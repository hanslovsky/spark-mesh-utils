package org.janelia.saalfeldlab.spark.mesh;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import gnu.trove.iterator.TLongLongIterator;
import gnu.trove.map.hash.TLongLongHashMap;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class WinnerTakesAllDownSampler
{

	public static void main( final String[] args ) throws IOException
	{
		final String inN5 = args[ 0 ];
		final String inDataset = args[ 1 ];
		final String outN5 = args[ 2 ];
		final String outGroup = args[ 3 ];
		final String scaleFactorsString = args[ 4 ];
		final String blockSizesString = args[ 5 ];

		final int[] scaleFactorsFlat = Arrays.stream( scaleFactorsString.split( "," ) ).mapToInt( Integer::parseInt ).toArray();
		final int[] blockSizesFlat = Arrays.stream( blockSizesString.split( "," ) ).mapToInt( Integer::parseInt ).toArray();

		final N5Reader n5in = BlockListsPerLabel.pathIsH5( inN5 ) ? new N5HDF5Reader( inN5 ) : new N5FSReader( inN5 );
		final DatasetAttributes inAttrs = n5in.getDatasetAttributes( inDataset );
		final int nDim = inAttrs.getNumDimensions();

		final int[][] scaleFactors = new int[ scaleFactorsFlat.length / nDim ][ nDim ];
		final int[][] blockSizes = new int[ blockSizesFlat.length / nDim ][ nDim ];

		for ( int i = 0, k = 0; i < scaleFactors.length; ++i, k += nDim )
			for ( int d = 0; d < nDim; ++d )
			{
				scaleFactors[ i ][ d ] = scaleFactorsFlat[ k + d ];
				blockSizes[ i ][ d ] = blockSizesFlat[ k + d ];
			}

		Paths.get( outN5, outGroup ).toFile().mkdirs();
		final Path target = Paths.get( outN5, outGroup, "s0" );
		target.toFile().delete();
		Files.createSymbolicLink( target, Paths.get( inN5, inDataset ) );

		final double[] resolution = Optional.ofNullable( n5in.getAttribute( inDataset, "resolution", double[].class ) ).orElse( DoubleStream.generate( () -> 1.0 ).limit( inAttrs.getNumDimensions() ).toArray() );

		final SparkConf conf = new SparkConf()
				.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
				.setAppName( MethodHandles.lookup().lookupClass().getName() );

		try (final JavaSparkContext sc = new JavaSparkContext( conf ))
		{

			for ( int level = 0, offByOne = 1; level < scaleFactors.length; ++level, ++offByOne )
			{
				final N5Reader outReader = BlockListsPerLabel.getReader( outN5, blockSizes[ Math.max( level - 1, 0 ) ] );
				final N5Writer outWriter = BlockListsPerLabel.getWriter( outN5, blockSizes[ level ] );

				final Path sourcePath = Paths.get( outGroup, String.format( "s%d", level ) );
				final Path targetPath = Paths.get( outGroup, String.format( "s%d", offByOne ) );

				final DatasetAttributes sourceAttrs = outReader.getDatasetAttributes( sourcePath.toString() );

				final int[] targetScaleFactor = scaleFactors[ level ];
				final int[] targetBlockSize = blockSizes[ level ];
				final int[] sourceBlockSize = blockSizes[ level == 0 ? level : level - 1 ];

				final long[] sourceDim = sourceAttrs.getDimensions();
				final long[] targetDim = IntStream.range( 0, nDim ).mapToLong( d -> ( long ) Math.ceil( sourceDim[ d ] * 1.0 / targetScaleFactor[ d ] ) ).toArray();
				System.out.println( "LOLWUT " + Arrays.toString( targetDim ) );
//				System.exit( 123 );

				outWriter.createDataset( targetPath.toString(), targetDim, targetBlockSize, sourceAttrs.getDataType(), sourceAttrs.getCompression() );
				outWriter.setAttribute( targetPath.toString(), "downsamplingFactors", targetScaleFactor );
				outWriter.setAttribute( targetPath.toString(), "resolution", resolution );

				for ( int d = 0; d < resolution.length; ++d )
					resolution[ d ] *= targetScaleFactor[ d ];

				final List< long[] > offsets = BlockListsPerLabel.collectAllOffsets( targetDim, targetBlockSize );

				final JavaRDD< long[] > offsetsRDD = sc.parallelize( offsets );
//				System.out.print( "GOT " + offsets.size() + " blocks." );
				downSample( sc, offsetsRDD, outN5, outN5, sourcePath.toString(), targetPath.toString(), targetScaleFactor, sourceBlockSize, targetBlockSize );

			}
		}

	}

	public static < I extends IntegerType< I > & NativeType< I > > void downSample(
			final JavaSparkContext sc,
			final JavaRDD< long[] > offsets,
			final String sourceGroup,
			final String targetGroup,
			final String sourcePath,
			final String targetPath,
			final int[] factors,
			final int[] sourceBlockSize,
			final int[] targetBlockSize )
	{
		offsets.map( offset -> {
			final N5Reader n5source = BlockListsPerLabel.getReader( sourceGroup, sourceBlockSize );
			final N5Writer n5target = BlockListsPerLabel.getWriter( targetGroup, targetBlockSize );

			final DatasetAttributes attrsTarget = n5source.getDatasetAttributes( targetPath );
			final long[] targetDims = attrsTarget.getDimensions();
			System.out.println( "ATTRS! " + Arrays.toString( attrsTarget.getDimensions() ) );
			final long[] max = new long[ targetDims.length ];
			for ( int d = 0; d < max.length; ++d )
			{
				final long stop = Math.min( offset[ d ] + targetBlockSize[ d ], targetDims[ d ] );
				max[ d ] = stop - 1;
			}

			final RandomAccessibleInterval< I > source = N5Utils.open( n5source, sourcePath );
			System.out.println( "1 DOING STUFF FOR " + Arrays.toString( offset ) + " " + Arrays.toString( max ) + " " + Arrays.toString( targetDims ) + " " + Arrays.toString( targetBlockSize ) );
			final Img< I > target = new ArrayImgFactory< I >().create( new FinalInterval( offset, max ), Util.getTypeFromInterval( source ).copy() );
			System.out.println( "2 DOING STUFF FOR " + Arrays.toString( offset ) + " " + Arrays.toString( max ) + " " + Arrays.toString( Intervals.dimensionsAsLongArray( target ) ) );
			final IntervalView< I > translatedTarget = Views.translate( target, offset );
			final I extension = Util.getTypeFromInterval( source ).copy();
			extension.setInteger( -1l );
			final RandomAccessible< I > extendedSource = Views.extendValue( source, extension );

			final long[] sourceMin = new long[ max.length ];
			final long[] sourceMax = new long[ max.length ];
			final TLongLongHashMap uniqueLabels = new TLongLongHashMap();
			for ( final Cursor< I > targetCursor = Views.flatIterable( translatedTarget ).localizingCursor(); targetCursor.hasNext(); )
			{
				final I t = targetCursor.next();
				targetCursor.localize( sourceMin );
				for ( int d = 0; d < sourceMin.length; ++d )
				{
					sourceMin[ d ] *= factors[ d ];
					sourceMax[ d ] = sourceMin[ d ] + factors[ d ];
				}

				uniqueLabels.clear();

				for ( final I s : Views.interval( extendedSource, new FinalInterval( sourceMin, sourceMax ) ) )
				{
					final long v = s.getIntegerLong();
					uniqueLabels.put( v, uniqueLabels.get( v ) + 1 );
				}

				long argMax = -1;
				long argMaxCount = -1;

				for ( final TLongLongIterator it = uniqueLabels.iterator(); it.hasNext(); )
				{
					it.advance();
					if ( it.value() > argMaxCount )
					{
						argMaxCount = it.value();
						argMax = it.key();
					}
				}

				t.setInteger( argMax );

			}

			final long[] gridOffset = IntStream.range( 0, max.length ).mapToLong( d -> offset[ d ] / targetBlockSize[ d ] ).toArray();
			N5Utils.saveBlock( translatedTarget, n5target, targetPath, gridOffset );
			System.out.println( "ATTRS AFTER! " + Arrays.toString( n5source.getDatasetAttributes( targetPath ).getDimensions() ) );

			return true;
		} ).count();
	}

}
