#include "cWatershed.h"

cWatershed::cWatershed() {}

cWatershed::~cWatershed() {}

stage1_obj cWatershed::stage_1(
    std::vector<float>& affinities,
    const unsigned int xsize,
    const unsigned int ysize,
    const unsigned int zsize,
    const float  high,
    const float  low,
    const std::size_t merge_size,
    const float merge_low,
    const std::size_t dust_size,
    const bool before_x,
    const bool after_x,
    const bool before_y,
    const bool after_y,
    const bool before_z,
    const bool after_z) {

    return this->run_chunk<float, unsigned int>(
        xsize, ysize, zsize, //sizes
        &affinities[0], //array to vector conversion
        high, low, merge_size, merge_low, dust_size,
        before_x, after_x, before_y, after_y, before_z, after_z);
}

template<class aff_type, class id_type>
stage1_obj cWatershed::run_chunk( 
    const std::size_t xsize,
    const std::size_t ysize,
    const std::size_t zsize,
    const aff_type* graph_flat,
    const aff_type  high,
    const aff_type  low,
    const std::size_t merge_size,
    const aff_type   merge_low,
    const std::size_t dust_size,
    const bool before_x,
    const bool after_x,
    const bool before_y,
    const bool after_y,
    const bool before_z,
    const bool after_z) {


    const std::size_t size = xsize*ysize*zsize;
    const index xdim = static_cast< index >( xsize );
    const index ydim = static_cast< index >( ysize );
    const index zdim = static_cast< index >( zsize );
 

    auto graph = this->initialize_graph<aff_type>(graph_flat, xdim, ydim, zdim);

    id_type* seg_flat = new id_type[size]; // TODO free this memory
    std::fill_n( seg_flat, size, 0 );
    auto seg = this->initialize_seg<id_type>(seg_flat, xdim, ydim, zdim);
    std::cout << "finished initializing" << std::endl;

    this->process_boundaries(graph, seg, xdim, ydim, zdim, high, before_x, after_x, before_y, after_y, before_z, after_z);
    std::cout << "finished process_boundaries" << std::endl;
    
    this->steepest_ascent(graph, seg, xdim, ydim, zdim, high, low);
    std::cout << "finished steepest_ascent" << std::endl;

    auto next_id = this->find_basins(seg_flat, seg, xdim, ydim, zdim, size);
    std::cout << "finished find_basins" << std::endl;

    auto counts = this->count_segments<id_type>(seg_flat, size, next_id);
    std::cout << "finished count_segments" << std::endl;

    auto region_graph = this->region_graph<id_type>(graph, seg, graph_flat, seg_flat, xdim, ydim, zdim, size, low, merge_low);
    std::cout << "finished region_graph" << std::endl;

    auto dendr = this->create_dendrogram<id_type, aff_type>(region_graph);
    std::cout << "finished create_dendrogram" << std::endl;

    this->count_borders(seg, counts, xdim, ydim, zdim, before_x, after_x, before_y, after_y, before_z, after_z);
    std::cout << "finished count_borders" << std::endl;

    auto sets = this->merge_regions<id_type, aff_type>(dendr, next_id, high, low, counts, merge_low, merge_size);
    std::cout << "finished merge_regions" << std::endl;

    id_type* remaps = this->relabel_seg<id_type>(seg_flat, counts, sets, dust_size, size);
    std::cout << "finished relabel_seg" << std::endl;

    this->relabel_dendogram<id_type, aff_type>(dendr, sets, remaps);
    std::cout << "finished relabel_dendogram" << std::endl;

    // use pybind11 to directly output tuples to python
    std::vector< unsigned int > dendr_u(dendr.size());
    std::vector< unsigned int > dendr_v(dendr.size());
    std::vector< float > dendr_aff(dendr.size());
    for ( std::size_t i = 0; i < dendr.size(); ++i )
    {
        dendr_u[i] = std::get<0>(dendr[i]);
        dendr_v[i] = std::get<1>(dendr[i]);
        dendr_aff[i] = std::get<2>(dendr[i]);
    }

    stage1_obj return_obj = { std::vector<unsigned int>(seg_flat, seg_flat + size),
                              dendr_u,
                              dendr_v,
                              dendr_aff,
                              counts };
    return return_obj;
};


template<class aff_type>
zi::watershed::const_volume_ref< aff_type, 4> cWatershed::initialize_graph(
    const aff_type* graph_flat, 
    const std::ptrdiff_t xdim,
    const std::ptrdiff_t ydim,
    const std::ptrdiff_t zdim) {

    zi::watershed::const_volume_ref< aff_type, 4 > graph( graph_flat, zi::watershed::extents[ xdim ][ ydim ][ zdim ][ 3 ] );
    return graph;
}

template<class id_type>
zi::watershed::volume_ref< id_type, 3 > cWatershed::initialize_seg(
    id_type* seg_flat, 
    const std::ptrdiff_t xdim,
    const std::ptrdiff_t ydim,
    const std::ptrdiff_t zdim) {

    zi::watershed::volume_ref< id_type, 3 > seg ( seg_flat,  zi::watershed::extents[ xdim ][ ydim ][ zdim ] );
    return seg;
}

template<class aff_type, class id_type>
void cWatershed::process_boundaries(
    zi::watershed::const_volume_ref< aff_type, 4 >& graph,
    zi::watershed::volume_ref< id_type, 3 >& seg,
    const std::ptrdiff_t xdim,
    const std::ptrdiff_t ydim,
    const std::ptrdiff_t zdim,
    const aff_type high,
    const bool before_x,
    const bool after_x,
    const bool before_y,
    const bool after_y,
    const bool before_z,
    const bool after_z) {

    const id_type front_x_border = before_x ? 0x0 : 0x08;
    const id_type back_x_border  = after_x  ? 0x0 : 0x01;
    const id_type front_y_border = before_y ? 0x0 : 0x10;
    const id_type back_y_border  = after_y  ? 0x0 : 0x02;
    const id_type front_z_border = before_z ? 0x0 : 0x20;
    const id_type back_z_border  = after_z  ? 0x0 : 0x04;

    for(index y = 1; y < ydim-1; ++y) {
        for(index z = 1; z < zdim-1; ++z) {
            seg[0][y][z]      = graph[1     ][y][z][0] >= high ? 0x08 : front_x_border;
            seg[xdim-1][y][z] = graph[xdim-1][y][z][0] >= high ? 0x01 : back_x_border;
        }
    }
    for(index x = 1; x < xdim-1; ++x) {
        for(index z = 1; z < zdim-1; ++z) {
            seg[x][0][z]      = graph[x][1     ][z][1] >= high ? 0x10 : front_y_border;
            seg[x][ydim-1][z] = graph[x][ydim-1][z][1] >= high ? 0x02 : back_y_border;
        }
    }
    for(index x = 1; x < xdim-1; ++x) {
        for(index y = 1; y < ydim-1; ++y) {
        
            seg[x][y][0]      = graph[x][y][1     ][2] >= high ? 0x20 : front_z_border;
            seg[x][y][zdim-1] = graph[x][y][zdim-1][2] >= high ? 0x04 : back_z_border;
        }
    }
}


template<class aff_type, class id_type>
void cWatershed::steepest_ascent(
    zi::watershed::const_volume_ref< aff_type, 4 >& graph,
    zi::watershed::volume_ref< id_type, 3 >& seg,
    const std::ptrdiff_t xdim,
    const std::ptrdiff_t ydim,
    const std::ptrdiff_t zdim,
    const aff_type high,
    const aff_type low) {
    /**
     Construct steepest ascent graph from affinity graph
  
    * `graph`: affinity graph (undirected and weighted). 4D array of affinities, where last dimension is of size 3
    * `seg`: steepest ascent graph (directed and unweighted). 
       `seg[x,y,z]` contains 6-bit number encoding edges outgoing from (x,y,z)
    * `low`: edges with affinity <= `low` are removed
    * `high`: affinities >= `high` are considered infinity
    Directed paths in the steepest ascent graph are steepest ascent paths
    in the affinity graph.  Both graphs are for 3D lattice with
    6-connectivity.  The steepest ascent graph can contain vertices with
    multiple outgoing edges if there are ties in the affinity graph, i.e.,
    if steepest ascent paths are nonunique.
    We follow the convention that:
    * `aff[x,y,z,1]` is affinity of voxels at [x-1,y,z] and [x,y,z]  
    * `aff[x,y,z,2]` is affinity of voxels at [x,y-1,z] and [x,y,z]  
    * `aff[x,y,z,3]` is affinity of voxels at [x,y,z-1] and [x,y,z]
    **/

    for(index x = 1; x < xdim-1; ++x) {
        for(index y = 1; y < ydim-1; ++y) {
            for(index z = 1; z < zdim-1; ++z) {
                const aff_type negx = graph[x][y][z][0];
                const aff_type negy = graph[x][y][z][1];
                const aff_type negz = graph[x][y][z][2];
                const aff_type posx = graph[x+1][y][z][0];
                const aff_type posy = graph[x][y+1][z][1];
                const aff_type posz = graph[x][y][z+1][2];
                const aff_type m = std::max({negx, negy, negz, posx, posy, posz});
                if ( m < low ) { continue; }    
                if ( negx == m || negx >= high ) { seg[x][y][z] |= 0x01; }
                if ( negy == m || negy >= high ) { seg[x][y][z] |= 0x02; }
                if ( negz == m || negz >= high ) { seg[x][y][z] |= 0x04; }
                if ( posx == m || posx >= high ) { seg[x][y][z] |= 0x08; }
                if ( posy == m || posy >= high ) { seg[x][y][z] |= 0x10; }
                if ( posz == m || posz >= high ) { seg[x][y][z] |= 0x20; }
           }
        }
    }

}

template<class id_type> 
id_type cWatershed::find_basins(
    id_type* seg_flat,
    zi::watershed::volume_ref< id_type, 3 >& seg,
    const std::ptrdiff_t xdim,
    const std::ptrdiff_t ydim,
    const std::ptrdiff_t zdim,
    std::size_t size) {
    //TODO divide this into two for loops as in 
    // https://github.com/seung-lab/watershed/blob/master/src-julia/divideplateaus.jl

    const index dindex[6]   = { ydim*zdim, zdim, 1, -ydim*zdim, -zdim, -1 }; //pox,posy,posz, negx, negy, negz
    const id_type dirmask[6]  = { 0x08, 0x10, 0x20, 0x01, 0x02, 0x04 };
    const id_type idirmask[6] = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20 };


    id_type next_id = 1;
    
    index* bfs       = zi::watershed::detail::get_buffer<index>(size+1); //allocates memory
    index  bfs_index = 0;
    index  bfs_start = 0;
    index  bfs_end   = 0;

    for(index ix = 1; ix < xdim-1; ++ix) {
        for(index iy = 1; iy < ydim-1; ++iy) {
            for(index iz = 1; iz < zdim-1; ++iz) {
                index idx = &seg[ix][iy][iz] - seg_flat;

                if ( !(seg_flat[ idx ] & assigned_bit ) )
                {
                    bfs[ bfs_end++ ] = idx;
                    seg_flat[ idx ] |= visited_bit;

                    while ( bfs_index != bfs_end )
                    {
                        index y = bfs[ bfs_index++ ];

                        for ( index d = 0; d < 6; ++d )
                        {
                            if ( seg_flat[ y ] & dirmask[ d ] )
                            {
                                index z = y + dindex[ d ];

                                if ( seg_flat[ z ] & assigned_bit )
                                {
                                    const id_type seg_id = seg_flat[ z ];
                                    while ( bfs_start != bfs_end )
                                    {
                                        seg_flat[ bfs[ bfs_start++ ] ] = seg_id;
                                    }
                                    bfs_index = bfs_end;
                                    d = 6; // (break)
                                }
                                else if ( !( seg_flat[ z ] & visited_bit ) )
                                {
                                    seg_flat[ z ] |= visited_bit;
                                    if ( !( seg_flat[ z ] & idirmask[ d ] ) )  // dfs now
                                    {
                                        bfs_index = bfs_end;
                                        d = 6; // (break)
                                    }
                                    bfs[ bfs_end++ ] = z;
                                }
                            }
                        }
                    }

                    if ( bfs_start != bfs_end )
                    {
                        while ( bfs_start != bfs_end )
                        {
                            seg_flat[ bfs[ bfs_start++ ] ] = assigned_bit | next_id;
                        }
                        ++next_id;
                    }
                }
                
            }
        }
    }
    zi::watershed::detail::return_buffer(bfs); //free memory
    
    return next_id;
}

template<class id_type>
std::vector< std::size_t > cWatershed::count_segments(
    id_type* seg_flat,
    std::size_t size,
    id_type& next_id){
    // if the highest bit of seg_flat[i] is set, we remove it, otherwise
    // we set seg_flat[i] to be zero. we avoid branching by using bit operations
    std::vector< std::size_t > counts;
    counts.resize(next_id);
    for ( id_type i = 0; i < static_cast< id_type >( size ); ++i ) {
        seg_flat[i] &= assigned_bit-(seg_flat[i]>>(std::numeric_limits<id_type>::digits-1));
        counts[ seg_flat[i] ] += 1;
    }
    return counts;
}

template<class id_type, class aff_type>
std::unordered_map< std::pair< id_type, id_type >, aff_type, pair_hash> cWatershed::region_graph(
    zi::watershed::const_volume_ref< aff_type, 4 >& graph,
    zi::watershed::volume_ref< id_type, 3 >& seg,
    const aff_type* graph_flat, 
    id_type* seg_flat,
    const std::ptrdiff_t xdim,
    const std::ptrdiff_t ydim,
    const std::ptrdiff_t zdim,
    std::size_t size,
    const aff_type low,
    const aff_type merge_low) {

    /*
    create region graph by finding maximum affinity between each pair of regions in segmentation
    * `graph`: affinity graph (undirected and weighted). 4D array of affinities, 
    where last dimension is of size 3
    * `seg`: segmentation.  Each element of the 3D array contains a *segment ID*,
    a nonnegative integer ranging from 0 to `max_segid`
    The vertices of the region graph are regions in the segmentation.  An
    edge of the region graph corresponds to a pair of regions in the
    segmentation that are connected by an edge in the affinity graph.  The
    weight of an edge in the region graph is the maximum weight of the
    edges in the affinity graph connecting the two regions.
    The region graph does not includes the edge between a region and itself.  
    Edges to background voxels (those with ID=0) are also ignored.
    */

    typedef std::pair< id_type, id_type >  id_pair;
    std::unordered_map<id_pair, aff_type, pair_hash> region_graph;
  
    aff_type dend_min = std::min( low, merge_low );
    const index rindex[3] = { -ydim*zdim, -zdim, -1 };

    for(index x = 1; x < xdim-1; ++x) {
        for(index y = 1; y < ydim-1; ++y) {
            for(index z = 1; z < zdim-1; ++z) {
                for ( index d = 0; d < 3; ++d ) {
        
                    index i = &seg[x][y][z] - seg_flat;
                    if ( graph[x][y][z][d] >= dend_min &&
                         seg[x][y][z] &&
                         seg_flat[ i + rindex[ d ] ] &&
                         seg_flat[ i + rindex[ d ] ] != seg_flat[ i ] )
                    {
                        id_pair p = std::minmax( seg_flat[ i ], seg_flat[ i + rindex[ d ] ] );
                        this->insert_or_update_edge<unsigned int, float>(region_graph, p, graph[x][y][z][d]);
                    }
                }
            }
        }
    }
    return region_graph;
}

template<class id_type, class aff_type>
std::vector< std::tuple< id_type, id_type, aff_type > > cWatershed::create_dendrogram(
    std::unordered_map< std::pair< id_type, id_type >, aff_type, pair_hash>& region_graph) {
    //This tranforms the unodered map into a vector so that it can be sorted by affinities
    //but I think this is not actually necesary because it shouldn't be required to do 
    //the merging in certain order.

    std::vector< std::tuple< id_type, id_type, aff_type > > dendr;   
    dendr.resize( region_graph.size() );
    index idx = 0;
    for (auto it : region_graph) {
        dendr[ idx ] = std::tie( it.first.first, it.first.second, it.second );
        ++idx;
    }
    // sort dendogram from high affinity to low affinity
    std::stable_sort( dendr.begin(), dendr.end(), zi::watershed::edge_compare< aff_type, id_type > );
    return dendr;
}

template<class id_type>
void cWatershed::count_borders(
    zi::watershed::volume_ref< id_type, 3 >& seg,
    std::vector< std::size_t >& counts,
    const std::ptrdiff_t xdim,
    const std::ptrdiff_t ydim,
    const std::ptrdiff_t zdim,
    const bool before_x,
    const bool after_x,
    const bool before_y,
    const bool after_y,
    const bool before_z,
    const bool after_z) {

    for(index y = 1; y < ydim-1; ++y) {
        for(index z = 1; z < zdim-1; ++z) {
            --counts[ seg[0][y][z] ];
            --counts[ seg[xdim-1][y][z] ];
            if ( before_x )
            {
                counts[ seg[0][y][z] ] |= on_border;
                counts[ seg[1][y][z] ] |= on_border;
            }

            if ( after_x )
            {
                counts[ seg[xdim-2][y][z] ] |= on_border;
                counts[ seg[xdim-1][y][z] ] |= on_border;
            }
        }
    }

    for(index x = 1; x < xdim-1; ++x) {
        for(index z = 1; z < zdim-1; ++z) {
            --counts[ seg[x][0][z] ];
            --counts[ seg[x][ydim-1][z] ];
            if ( before_y )
            {
                counts[ seg[x][0][z] ] |= on_border;
                counts[ seg[x][1][z] ] |= on_border;
            }

            if ( after_y )
            {
                counts[ seg[x][ydim-2][z] ] |= on_border;
                counts[ seg[x][ydim-1][z] ] |= on_border;
            }
        }
    }

    for(index x = 1; x < xdim-1; ++x) {
        for(index y = 1; y < ydim-1; ++y) {
            --counts[ seg[x][y][0] ];
            --counts[ seg[x][y][zdim-1] ];
            if ( before_z )
            {
                counts[ seg[x][y][0] ] |= on_border;
                counts[ seg[x][y][1] ] |= on_border;
            }

            if ( after_z )
            {
                counts[ seg[x][y][zdim-2] ] |= on_border;
                counts[ seg[x][y][zdim-1] ] |= on_border;
            }
        }
    }
    counts[0] = 0;
}

template<class id_type, class aff_type>
zi::disjoint_sets< id_type > cWatershed::merge_regions(
    std::vector< std::tuple< id_type, id_type, aff_type > >& dendr,
    id_type& next_id,
    const aff_type high,
    const aff_type low,
    std::vector< std::size_t >& counts,
    const aff_type merge_low,
    const std::size_t merge_size
    ) {
    /* Agglomerative clustering proceeds by considering the edges 
       of the region graph in sequence. 
       If either region has size less than `merge_size`, then merge the regions.
       When the weight of the edge in the region graph is 
       less than or equal to `merge_low`, agglomeration terminates.

       TODO explain border handling.
    */

    zi::disjoint_sets< id_type > sets( next_id );
    for (std::size_t i = 0; i < dendr.size(); ++i )
    {
        id_type v1 = sets[std::get<0>(dendr[i])]; // root of dendr[i][0]
        id_type v2 = sets[std::get<1>(dendr[i])]; // root of dendr[i][1]
        aff_type aff_value =  std::get<2>(dendr[i]);
        if ( aff_value >= high )
        {
            if ( v1 != v2 ) { //If roots are not the same merge them
                counts[v1] += counts[v2]&(~on_border); //removes flag from v2
                counts[v1] |= counts[v2]&on_border; //reapplys flag if either v1 or v2 had it
                counts[v2]  = 0;
                std::swap( counts[ sets.join(v1,v2) ], counts[v1] );
                --next_id;
            }
        
        }
        else if ( aff_value >= merge_low ) {
            if ( v1 != v2 )
            {
                if ((counts[v1]<merge_size) || (counts[v2]<merge_size))
                {
                    if ((on_border&(counts[v1]|counts[v2]))==0)
                    {
                        counts[v1] += counts[v2];
                        counts[v2]  = 0;
                        std::swap( counts[ sets.join(v1,v2) ], counts[v1] );
                        --next_id;
                    }
                    else
                    {
                        counts[v1] |= counts[v2]&on_border;
                        counts[v2] |= counts[v1]&on_border;
                    }
                }
            }
        }
        else if ( aff_value < low ) { break; }
    }
    return sets;
}

template<class id_type>
id_type* cWatershed::relabel_seg(
    id_type* seg_flat,
    std::vector< std::size_t >& counts, 
    zi::disjoint_sets< id_type >& sets,
    const std::size_t dust_size,
    std::size_t size) {

    id_type* remaps = zi::watershed::detail::get_buffer<id_type>(counts.size());
    remaps[0] = 0;
    counts[0] &= ~on_border;
    id_type next_new_id = 1;
    for (std::size_t i = 1; i < counts.size(); ++i ) {
        if ( !(counts[i]&(~on_border)) ) { continue; }
        if ( counts[i] >= dust_size ) {
            counts[next_new_id] = counts[i]&(~on_border);
            remaps[i] = next_new_id;
            ++next_new_id;
        }
        else { //setting to black
            counts[0] += counts[i]&(~on_border);
            counts[i]  = remaps[i] = 0;
        }
    }

    for ( std::size_t i = 1; i < counts.size(); ++i ) {
        remaps[i] = remaps[sets[i]];
    }

    for ( std::size_t i = 0; i < size; ++i ) {
        seg_flat[ i ] = remaps[ seg_flat[ i ] ];
    }

    counts.resize( next_new_id );
    sets.resize( next_new_id );
    return remaps;
}

template<class id_type, class aff_type>
void cWatershed::relabel_dendogram(
    std::vector< std::tuple< id_type, id_type, aff_type > >& dendr,
    zi::disjoint_sets< id_type >& sets,
    id_type* remaps) {

    index dend_len = 0;
    for ( std::size_t i = 0; i < dendr.size(); ++i )
    {
        const id_type a1 = remaps[ std::get<0>(dendr[i]) ];
        const id_type a2 = remaps[ std::get<1>(dendr[i]) ];
        const id_type v1 = sets[a1];
        const id_type v2 = sets[a2];
        if ( v1 && v2 && v1 != v2 ) {
            sets.join( v1, v2 );
            dendr[ dend_len++ ] = std::tie( a1, a2, std::get<2>( dendr[ i ] ) );
            assert (a1 != 0);
            assert (a2 != 0); 
        }
    }

    zi::watershed::detail::return_buffer(remaps);
    dendr.resize( dend_len );
}

std::vector<unsigned int> cWatershed::stage_2(
    std::vector<unsigned int>& seg_border_front,
    std::vector<unsigned int>& seg_border_back,
    std::vector<float>& aff_border,
    std::vector<unsigned int>& u_front,
    std::vector<unsigned int>& v_front,
    std::vector<float>& aff_front,
    std::vector<unsigned int>& u_back,
    std::vector<unsigned int>& v_back,
    std::vector<float>& aff_back,
    std::vector<std::size_t>& segment_sizes_front,
    std::vector<std::size_t>& segment_sizes_back) {

 
    float high=0.999; 
    float low=0.006821;
    unsigned int merge_size=800;
    float merge_low=0.3;
    unsigned int dust_size=100;

    const unsigned int front_id_offset = 0;
    const unsigned int back_id_offset =  segment_sizes_front.size();
    const std::size_t total_num_domains = front_id_offset + back_id_offset;

    assert (seg_border_front.size() == seg_border_back.size());
    assert (seg_border_front.size() == aff_border.size());

    std::unordered_map< id_pair, float, pair_hash> edge_map;  
    for ( std::size_t i = 0; i < seg_border_front.size(); ++i ) {
        if ( !seg_border_front[i] || !seg_border_back[i] ){ continue; } // check the outers are not background
        id_pair edge_ids( front_id_offset+seg_border_front[i], back_id_offset+seg_border_back[i] );
        if ( aff_border[i] >= low ) {
            this->insert_or_update_edge<unsigned int, float>(edge_map, edge_ids, aff_border[i]);
        }
    }

    //merge sizes
    std::vector<std::size_t> sizes(segment_sizes_front.size()+ segment_sizes_back.size());
    for ( std::size_t i = 0; i < segment_sizes_front.size(); ++i ) {
        sizes[i] = segment_sizes_front[i];
    }
    for ( std::size_t i = 0; i < segment_sizes_back.size(); ++i ) {
        sizes[i+back_id_offset] = segment_sizes_back[i];
    } 


    //merge all dendograms
    std::vector< std::tuple< unsigned int, unsigned int, float > > dendr;   
    for (auto it : edge_map) {
        dendr.push_back(std::tie( it.first.first, it.first.second, it.second ));
    }


    for ( std::size_t i = 0; i < u_front.size(); ++i ) {
        const unsigned int u = u_front[i] +  front_id_offset;
        const unsigned int v = u_front[i] +  front_id_offset;
        dendr.push_back(std::tie(u, v,  aff_front[i] ));
    }
    for ( std::size_t i = 0; i < u_back.size(); ++i ) {
        const unsigned int u = u_back[i] + back_id_offset;
        const unsigned int v = v_back[i] + back_id_offset;
        dendr.push_back(std::tie(u, v, aff_back[i]));
    }

    // sort dendogram from high affinity to low affinity
    std::stable_sort( dendr.begin(), dendr.end(), zi::watershed::edge_compare<float,unsigned int> );
    zi::disjoint_sets<unsigned int> sets(segment_sizes_front.size() + segment_sizes_back.size());

    for ( std::size_t i = 0; i < dendr.size(); ++i ) {

        const unsigned int v1  = sets.find_set( std::get<0>(dendr[i]) );
        const unsigned int v2  = sets.find_set( std::get<1>(dendr[i]) );
        const float aff  = std::get<2>(dendr[i]);
        if ( aff < low ) { break; }

        if ( v1 == v2 ) { continue; }
        if ( aff > high ) {    
            // std::cout << " merging u: " << std::get<0>(dendr[i]) << " v:" << std::get<1>(dendr[i]) 
            // << " aff: " << aff  << " low " << low << std::endl;
            sizes[v1] += sizes[v2];
            sizes[v2]  = 0;
            std::swap( sizes[ sets.join(v1,v2) ], sizes[v1] );
        }

    }


    for ( std::size_t i = 0; i < dendr.size(); ++i ) {

        const unsigned int v1  = sets.find_set( std::get<0>(dendr[i]) );
        const unsigned int v2  = sets.find_set( std::get<1>(dendr[i]) );
        const float aff  = std::get<2>(dendr[i]);
        if ( aff < low ) { break; }

        if ( v1 == v2 ) { continue; }
        if ( sizes[v1] < merge_size || sizes[v2] < merge_size ) {
            // std::cout << "merge_size merging u: " << std::get<0>(dendr[i]) << " v:" << std::get<1>(dendr[i]) 
            // << " aff: " << aff << std::endl;

            sizes[v1] += sizes[v2];
            sizes[v2]  = 0;
            std::swap( sizes[ sets.join(v1,v2) ], sizes[v1] );
        }
    }  

    std::cout << "\n Renumber the domains" << std::endl;
    std::vector<unsigned int> remaps(total_num_domains);
    remaps[0] = 0;
    unsigned int new_index = 1;
    for ( unsigned int i = 1; i < total_num_domains; ++i )
    {

        if ( sizes[ i ] >= dust_size ) {
            sizes[ new_index ] = sizes[ i ];
            remaps[ i ] = new_index;
            ++new_index;
        }
        else {
            sizes[ 0 ] += sizes[ i ];
            sizes[ i ]  = remaps[ i ] = 0;
        }

    }

    // for ( unsigned int i = 1; i < total_num_domains; ++i )
    // {
    //     if ( !remaps[ i ] )
    //     {
    //         remaps[ i ] = remaps[ sets.find_set( i ) ];
    //     }
    //     // std::cout << i << " -> " << remaps[i] << std::endl;
    // }

    return remaps;

}

template<class id_type, class aff_type>
void cWatershed::insert_or_update_edge( std::unordered_map< std::pair< id_type, id_type >, aff_type, pair_hash>& map, 
                                        std::pair< id_type, id_type >& pair,
                                        aff_type aff) {

    auto it = map.find(pair);
    if ( it == map.end() ) { map.insert( std::make_pair( pair, aff )); }
    else if( it->second < aff ) {  it->second = aff; }
}

std::vector<unsigned int> cWatershed::stage_3(std::vector<unsigned int>& seg,
                         std::vector<unsigned int>& mapping) {
    for ( std::size_t i = 0; i < seg.size(); ++i ) {
        seg[i] = mapping[ seg[i] ];
    }
    return seg;
}

