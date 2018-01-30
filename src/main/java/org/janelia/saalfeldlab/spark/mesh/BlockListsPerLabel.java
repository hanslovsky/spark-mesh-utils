package org.janelia.saalfeldlab.spark.mesh;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import gnu.trove.list.array.TLongArrayList;
import gnu.trove.set.hash.TLongHashSet;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import scala.Tuple2;

public class BlockListsPerLabel
{

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static long INVALID = -1;

	public static void main( final String[] args ) throws IOException
	{
		final String n5BasePath = args[ 0 ];
		final String dataset = args[ 1 ];
		final String target = args[ 2 ];

		new File( target ).mkdirs();

		final N5Reader n5 = pathIsH5( n5BasePath ) ? new N5HDF5Reader( n5BasePath ) : new N5FSReader( n5BasePath );
		final DatasetAttributes attrs = n5.getDatasetAttributes( dataset );
		final long[] dims = attrs.getDimensions();
		final int nDim = attrs.getNumDimensions();
		final int[] blockSize = attrs.getBlockSize();
		final List< Interval > intervals = collectAllOffsets( dims, blockSize, min -> new FinalInterval( min, IntStream.range( 0, nDim ).mapToLong( d -> Math.min( min[ d ] + blockSize[ d ], dims[ d ] ) - 1 ).toArray() ) );

		final SparkConf conf = new SparkConf();
		conf
				.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
				.setAppName( MethodHandles.lookup().lookupClass().getName() );

		try (final JavaSparkContext sc = new JavaSparkContext( conf ))
		{

			final JavaRDD< Tuple2< Interval, long[] > > ids = collectIds( sc.parallelize( intervals ), n5BasePath, dataset );

			ids
					.flatMapToPair( t -> Arrays.stream( t._2() ).mapToObj( l -> new Tuple2<>( l, t._1() ) ).iterator() )
					.combineByKey( BlockListsPerLabel::toArrayList,
							( list, value ) -> {
								list.addAll( toArrayList( value ) );
								return list;
							}, ( l1, l2 ) -> {
								l1.addAll( l2 );
								return l1;
							} )
					.map( idAndLocations -> {
						final Long id = idAndLocations._1();
						final TLongArrayList locations = idAndLocations._2();
						final byte[] data = new byte[ locations.size() * Long.BYTES ];
						final File f = new File( target, String.format( "%d", id ) );
						f.createNewFile();
						final ByteBuffer bb = ByteBuffer.wrap( data );
						for ( int i = 0; i < locations.size(); ++i )
							bb.putLong( locations.get( i ) );

						try (FileOutputStream fos = new FileOutputStream( f ))
						{
							LOG.debug( "Saving {} locations for label {}: {}.", locations.size(), id, locations );
							fos.write( data );
						}
						return true;
					} )
					.count();
		}

	}

	public static < I extends IntegerType< I > & NativeType< I > > JavaRDD< Tuple2< Interval, long[] > > collectIds(
			final JavaRDD< Interval > intervalsRDD,
			final String n5BasePath,
			final String dataset )
	{
		return intervalsRDD.map( interval -> {
			final N5Reader reader = pathIsH5( n5BasePath ) ? new N5HDF5Reader( n5BasePath ) : new N5FSReader( n5BasePath );
			final RandomAccessibleInterval< I > data = N5Utils.open( reader, dataset );

			final TLongHashSet labels = new TLongHashSet();
			for ( final I l : Views.interval( data, interval ) )
			{
				final long v = l.getIntegerLong();
				// TODO for now check for larger than zero -- better check in
				// the future (from arguments)
				if ( v > 0 )
					labels.add( v );
			}

			return new Tuple2<>( interval, labels.toArray() );
		} );
	}

	public static boolean pathIsH5( final String path )
	{
		return Pattern.matches( "^.*\\.h5$|^.*\\.hdf$", path );
	}

	public static List< long[] > collectAllOffsets( final long[] dimensions, final int[] blockSize )
	{
		return collectAllOffsets( dimensions, blockSize, block -> block );
	}

	public static < T > List< T > collectAllOffsets( final long[] dimensions, final int[] blockSize, final Function< long[], T > func )
	{
		return collectAllOffsets( new long[ dimensions.length ], Arrays.stream( dimensions ).map( d -> d - 1 ).toArray(), blockSize, func );
	}

	public static List< long[] > collectAllOffsets( final long[] min, final long[] max, final int[] blockSize )
	{
		return collectAllOffsets( min, max, blockSize, block -> block );
	}

	public static < T > List< T > collectAllOffsets( final long[] min, final long[] max, final int[] blockSize, final Function< long[], T > func )
	{
		final List< T > blocks = new ArrayList<>();
		final int nDim = min.length;
		final long[] offset = min.clone();
		for ( int d = 0; d < nDim; )
		{
			final long[] target = offset.clone();
			blocks.add( func.apply( target ) );
			for ( d = 0; d < nDim; ++d )
			{
				offset[ d ] += blockSize[ d ];
				if ( offset[ d ] <= max[ d ] )
					break;
				else
					offset[ d ] = 0;
			}
		}
		return blocks;
	}

	public static TLongArrayList toArrayList( final Interval interval )
	{
		final TLongArrayList al = new TLongArrayList();
		al.addAll( Intervals.minAsLongArray( interval ) );
		al.addAll( Intervals.maxAsLongArray( interval ) );
		return al;
	}

}
