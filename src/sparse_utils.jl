function print_sparse_statistics(S)
    println("size: ", size(S), "; rank: ", rank(S))
    println("sparsity: ", nnz(S)/length(S))
    println("memory: ", Base.summarysize(S))
end