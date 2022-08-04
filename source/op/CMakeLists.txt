
file(GLOB OP_SRC prod_force.cpp prod_env_mat.cpp)
# file(GLOB OP_GRADS_SRC )

# Define our library target
add_library(op_abi SHARED ${OP_SRC})
# add_library(op_grads SHARED ${OP_GRADS_SRC})

message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message("Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(op_abi PUBLIC ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
# include_directories(op_grads ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
# include_directories(op_grads ${TORCH_INCLUDE_DIRS})

target_link_libraries(op_abi PRIVATE ${LIB_DEEPMD} ${TORCH_LIBRARIES} ${Python3_LIBRARIES})
# target_link_libraries(op_grads PRIVATE ${LIB_DEEPMD})

# Link against LibTorch
# target_link_libraries(op_abi )
# target_link_libraries(op_grads "${TORCH_LIBRARIES}")
