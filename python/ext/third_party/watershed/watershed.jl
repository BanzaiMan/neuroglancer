# load dependencies
include("steepestascent.jl")
include("divideplateaus.jl")
include("findbasins.jl")
include("regiongraph.jl")
include("mergeregions.jl")
include("mst.jl")

using HDF5
input_path = ARGS[1]
print("input path: ")
print(input_path)

output_path = ARGS[2]
print("\noutput path: ")
print(output_path)

high_threshold = parse(Float32,ARGS[3])
print("\nhigh_threshold: ")
print(high_threshold)

low_threshold = parse(Float32,ARGS[4])
print("\nlow: ")
print(low_threshold)

merge_threshold = parse(Float32,ARGS[5])
print("\nmerge_threshold: ")
print(merge_threshold)

merge_size = parse(UInt32,ARGS[6])
print("\nmerge_size: ")
print(merge_size)

dust_size = parse(UInt32,ARGS[7])
print("\ndust_size(this is currently ignored):")
print(dust_size)
print("\n")

# read affinities
aff = h5read(input_path,"main");
# created steepest ascent graph
sag=steepestascent(aff,low_threshold,high_threshold);
# divided plateaus
divideplateaus!(sag);
# find basins
(seg, counts, counts0) = findbasins(sag);
# create region graph
rg = regiongraph(aff,seg,length(counts));
# merge regions
new_rg = mergeregions(seg,rg,counts,[(merge_size,merge_threshold)]);
# create maximal spanning tree
rt = mst(new_rg,length(counts));

# export
#TODO export mst
h5open(output_path, "w") do file
    write(file, "/main", seg)
end