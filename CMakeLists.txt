cmake_minimum_required(VERSION 3.10)
project(GP-ANNS)

# for converting relative paths to absolute paths
cmake_policy(SET CMP0076 NEW)

option(KAMINPAR_64BIT_WEIGHTS "Enable 64 bit weights in KaMinPar" ON)
option(KAMINPAR_64BIT_EDGE_IDS "Enable 64 bit edge IDs in KaMinPar" ON)
option(KAMINPAR_ENABLE_TIMERS "Enable KaMinPar running time measurements (disable to call the library in parallel)" OFF)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wunused")
set(CMAKE_CXX_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=native -march=native -ffast-math")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lm")
#set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}")

add_subdirectory(external/parlaylib)
add_subdirectory(external/KaMinPar)
add_subdirectory(external/hnswlib)
# add_subdirectory(external/message-queue)

include_directories("src")

add_executable(SmallScaleQueries small_scale_queries.cpp)
add_executable(Convert convert.cpp)
add_executable(Partition partition.cpp)
add_executable(QueryAttribution run_query_attribution.cpp)
add_executable(OracleRecall oracle_recall.cpp)
# add_executable(DistributedBench distributed_bench.cpp)
add_executable(GraphQualityBench graph_quality_benchmark.cpp)
add_executable(AnalyzeApproximationLosses analyze_approximation_losses.cpp)
add_executable(NaiveInsertBench naive_insert_bench.cpp)
add_executable(BuildRouter build_router.cpp)

set(TARGETS
		SmallScaleQueries
		Convert
		Partition
		QueryAttribution
		OracleRecall
		# DistributedBench
		GraphQualityBench
		AnalyzeApproximationLosses
		NaiveInsertBench
		BuildRouter
)

foreach(target IN LISTS TARGETS)
	target_link_libraries(${target} PUBLIC parlay)
endforeach()

target_link_libraries(Partition PUBLIC kaminpar_shm)
target_link_libraries(GraphQualityBench PUBLIC kaminpar_shm)
# target_link_libraries(DistributedBench PUBLIC message-queue kassert::kassert)

OPTION(MIPS_DISTANCE "Use MIPS distance instead of L2" OFF)
if(MIPS_DISTANCE)
	foreach(target IN LISTS TARGETS)
		target_compile_definitions(${target} PUBLIC MIPS_DISTANCE)
	endforeach()
endif(MIPS_DISTANCE)

find_library(TCMALLOC_LIB tcmalloc)
if (TCMALLOC_LIB)
	target_link_libraries(Partition PUBLIC ${TCMALLOC_LIB})
	# target_link_libraries(QueryAttribution PUBLIC ${TCMALLOC_LIB})
	message(STATUS "Using tcmalloc: ${TCMALLOC_LIB}")
else ()
	message(STATUS "tcmalloc enabled but unavailable on this system")
endif ()


add_subdirectory(src)

find_package(MPI)
if(MPI_CXX_FOUND)
    add_executable(DistributedBench distributed_bench.cpp)
    target_link_libraries(DistributedBench PUBLIC parlay MPI::MPI_CXX)
    if(MIPS_DISTANCE)
        target_compile_definitions(DistributedBench PUBLIC MIPS_DISTANCE)
    endif()
    target_sources(DistributedBench PRIVATE src/kmeans_tree_router.cpp
                                            src/defs.cpp src/dist.cpp
                                            src/kmeans.cpp src/points_io.cpp
                                            src/metis_io.cpp)

    add_executable(DistributedInsertBench distributed_insert_bench.cpp)
    target_link_libraries(DistributedInsertBench PUBLIC parlay MPI::MPI_CXX)
    if(MIPS_DISTANCE)
        target_compile_definitions(DistributedInsertBench PUBLIC MIPS_DISTANCE)
    endif()
    target_sources(DistributedInsertBench PRIVATE src/kmeans_tree_router.cpp
                                                   src/defs.cpp src/dist.cpp
                                                   src/kmeans.cpp src/points_io.cpp
                                                   src/metis_io.cpp)
    add_executable(DistributedInsertBenchSweep distributed_insert_bench_sweep.cpp)
    target_link_libraries(DistributedInsertBenchSweep PUBLIC parlay MPI::MPI_CXX)
    if(MIPS_DISTANCE)
        target_compile_definitions(DistributedInsertBenchSweep PUBLIC MIPS_DISTANCE)
    endif()
    target_sources(DistributedInsertBenchSweep PRIVATE src/kmeans_tree_router.cpp
                                                        src/defs.cpp src/dist.cpp
                                                        src/kmeans.cpp src/points_io.cpp
                                                        src/metis_io.cpp)
    message(STATUS "MPI found — DistributedBench + DistributedInsertBench + DistributedInsertBenchSweep will be built")
else()
    message(STATUS "MPI not found — MPI targets will NOT be built")
endif()