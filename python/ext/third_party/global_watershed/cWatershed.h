#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <zi/disjoint_sets/disjoint_sets.hpp> 

#include <detail/volume_ref.hpp>
#include <detail/buffer.hpp>
#include <detail/utility.hpp>


template <class T> //taken from boost
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        std::size_t seed = 0;
        hash_combine(seed, h1);
        hash_combine(seed, h2);
        return seed;  
    }
};


struct stage1_obj
{
    std::vector<unsigned int> seg;
    std::vector< unsigned int > dendr_u;
    std::vector< unsigned int > dendr_v;
    std::vector< float > dendr_aff;
    std::vector< std::size_t > segment_sizes;
};

class cWatershed {
public:
  cWatershed();
  ~cWatershed();

  stage1_obj stage_1(
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
    const bool after_z);

    std::vector<unsigned int>  stage_2(std::vector<unsigned int>& seg_border_front,
               std::vector<unsigned int>& seg_border_back,
               std::vector<float>& aff_border,
               std::vector<unsigned int>& u_front,
               std::vector<unsigned int>& v_front,
               std::vector<float>& aff_front,
               std::vector<unsigned int>& u_back,
               std::vector<unsigned int>& v_back,
               std::vector<float>& aff_back,
               std::vector<std::size_t>& segment_sizes_front,
               std::vector<std::size_t>& segment_sizes_back);

    std::vector<unsigned int> stage_3(std::vector<unsigned int>& seg,
                 std::vector<unsigned int>& mapping);
private:

    static const unsigned int assigned_bit = (1ul<<(std::numeric_limits<unsigned int>::digits-1));
    static const unsigned int visited_bit = 0x40;
    static const std::size_t on_border = (1ul<<(std::numeric_limits<std::size_t>::digits-1));
    typedef std::ptrdiff_t index;
    typedef std::pair< unsigned int, unsigned int > id_pair;

    template<class aff_type, class id_type>
    stage1_obj run_chunk(
        const std::size_t xsize,
        const std::size_t ysize,
        const std::size_t zsize,
        const aff_type* aff,
        const aff_type  high,
        const aff_type  low,
        const std::size_t merge_size,
        const aff_type merge_low,
        const std::size_t dust_size,
        const bool before_x,
        const bool after_x,
        const bool before_y,
        const bool after_y,
        const bool before_z,
        const bool after_z);

    template<class aff_type>
    zi::watershed::const_volume_ref< aff_type, 4 > initialize_graph(
        const aff_type* graph_flat, 
        const std::ptrdiff_t xdim,
        const std::ptrdiff_t ydim,
        const std::ptrdiff_t zdim);

    template<class id_type>
    zi::watershed::volume_ref< id_type, 3 > initialize_seg(
        id_type* seg_flat, 
        const std::ptrdiff_t xdim,
        const std::ptrdiff_t ydim,
        const std::ptrdiff_t zdim);

    template<class aff_type, class id_type>
    void process_boundaries(
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
        const bool after_z);

    template<class aff_type, class id_type>
    void steepest_ascent(
        zi::watershed::const_volume_ref< aff_type, 4 >& graph,
        zi::watershed::volume_ref< id_type, 3 >& seg,
        const std::ptrdiff_t xdim,
        const std::ptrdiff_t ydim,
        const std::ptrdiff_t zdim,
        const aff_type high,
        const aff_type low);

    template<class id_type>
    id_type find_basins(
        id_type* seg_flat,
        zi::watershed::volume_ref< id_type, 3 >& seg,
        const std::ptrdiff_t xdim,
        const std::ptrdiff_t ydim,
        const std::ptrdiff_t zdim,
        const std::size_t size);

    template<class id_type>
    std::vector< std::size_t > count_segments(
        id_type* seg_flat,
        const std::size_t size,
        id_type& next_id);

    template<class id_type, class aff_type>
    std::unordered_map< std::pair< id_type, id_type >, aff_type, pair_hash> region_graph(
        zi::watershed::const_volume_ref< aff_type, 4 >& graph,
        zi::watershed::volume_ref< id_type, 3 >& seg,
        const aff_type* graph_flat, 
        id_type* seg_flat,
        const std::ptrdiff_t xdim,
        const std::ptrdiff_t ydim,
        const std::ptrdiff_t zdim,
        const std::size_t size,
        const aff_type low,
        const aff_type merge_low);

    template<class id_type, class aff_type>
    std::vector< std::tuple< id_type, id_type, aff_type > > create_dendrogram(
        std::unordered_map< std::pair< id_type, id_type >, aff_type, pair_hash>& max_affinity_map);

    template<class id_type>
    void count_borders(
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
        const bool after_z);

    template<class id_type,class aff_type>
    zi::disjoint_sets< id_type > merge_regions(
        std::vector< std::tuple< id_type, id_type, aff_type > >& dendr,
        id_type& next_id,
        const aff_type high,
        const aff_type low,
        std::vector< std::size_t >& counts,
        const aff_type merge_low,
        const std::size_t merge_size);

    template<class id_type>
    id_type* relabel_seg(
        id_type* seg_flat,
        std::vector< std::size_t >& counts, 
        zi::disjoint_sets< id_type >& sets,
        const std::size_t dust_size,
        const std::size_t size);

    template<class id_type, class aff_type>
    void relabel_dendogram(
        std::vector< std::tuple< id_type, id_type, aff_type > >& dendr,
        zi::disjoint_sets< id_type >& sets,
        id_type* remaps);

    template<class id_type, class aff_type>
    void insert_or_update_edge( std::unordered_map< std::pair< id_type, id_type >, aff_type, pair_hash>& map, 
                                std::pair< id_type, id_type >& pair,
                                aff_type aff);
};
