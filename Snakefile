configfile: "config.yaml"

rule link_bam:
    input:
        src=lambda wildcards: "%s/{sample}.bam" % config["input_bam_dir"]
    output:
        "data/input_alignments/{sample}.bam"
    threads:
        1
    shell:
        "ln -s {input.src} {output}"


# filter bams
rule index_bam:
    input:
        "data/input_alignments/{sample}.bam"
    output:
        "data/input_alignments/{sample}.bai"
    threads:
        4
    shell:
        "samtools index -@{threads} {input} {output}"
        
rule filter_bam:
    input:
        bam="data/input_alignments/{sample}.bam",
        bai="data/input_alignments/{sample}.bai"
    params:
        min_mapping_qual=config["min_mapping_quality"],
        chrom=config["chrom"]
    output:
        "data/filtered_alignments/{sample}.filtered.bam"
    threads:
        4
    shell:
        "samtools view -b -@{threads} -F 0x0200 -F 0x0100 -F 0x004 -q {params.min_mapping_qual} {input.bam} {params.chrom} > {output}"

# k-mer counting with jellyfish
rule extract_bam_reads:
    input:
        bam="data/filtered_alignments/{sample}.filtered.bam"
    params:
        kmer_size=config["kmer_size"]
    output:
        jf="data/kmer_counts/{sample}.jf"
    threads:
        6
    shell:
        "samtools fasta {input.bam} | jellyfish count -t {threads} -m {params.kmer_size} -s 1000M -C -o {output.jf} /dev/fd/0"

# jellyfish dump is single threaded
# but specify number of threads to avoid
# overwhelming I/O
rule dump_kmers:
    input:
        jf="data/kmer_counts/{sample}.jf"
    output:
        counts="data/kmer_counts/{sample}.counts"
    threads:
        6
    shell:
        "jellyfish dump -c -o {output.counts} {input.jf}"

# find document frequencies
# note that pre-allocating a buffer (-S) makes
# a massive difference in run time
rule sort_kmers:
    input:
        counts="data/kmer_counts/{sample}.counts"
    output:
        sorted_kmers="data/doc_freq/{sample}.sorted.counts"
    threads:
        6
    shell:
        "cut -f 1 -d \" \" {input.counts} | sort -S 16G --parallel {threads} > {output.sorted_kmers}"

# merge operation doesn't actually use
# multiple threads but using all threads
# lets us prevent other processes from running
# so that we can use all memory as buffer
rule create_pass_list:
    input:
        sorted_kmers=expand("data/doc_freq/{sample}.sorted.counts",
                            sample=config["samples"])
    params:
        batch_size=config["merge_batch_size"],
        min_df=config["min_doc_freq"]
    output:
        pass_list="data/doc_freq/kmer_passlist.bf"
    threads:
        24
    shell:
        "sort --batch-size {params.batch_size} -S 56G -T data/doc_freq/ -m {input.sorted_kmers} | uniq -c | scripts/create_bloomfilter.py --min-df {params.min_df} --output-bf {output.pass_list}"

# feature extraction
rule create_random_projection:
    params:
        n_features=config["n_features"],
        n_rand_dim=config["n_rand_dim"],
        n_samples=len(config["samples"])
    output:
        rand_proj="data/feature_extraction/rand_proj.joblib"
    threads:
        4
    shell:
        "scripts/create_rand_proj.py --n-features {params.n_features} --n-components {params.n_rand_dim} --n-samples {params.n_samples} --output-fl {output.rand_proj}"

rule extract_features:
    input:
        kmer_counts="data/kmer_counts/{sample}.counts",
        rand_proj="data/feature_extraction/rand_proj.joblib",
        pass_list="data/doc_freq/kmer_passlist.bf"
    params:
        n_features=config["n_features"],
        use_binary=config["use_binary_features"]
    output:
        feature_matrix="data/feature_extraction/{sample}.features"
    threads:
        3
    shell:
        "scripts/feature_extractor.py --rand-proj-fl {input.rand_proj} --passlist-bf {input.pass_list} --kmer-freq-fl {input.kmer_counts} --n-features {params.n_features} {params.use_binary} --feature-matrix {output.feature_matrix}"

rule pca:
    input:
        feature_matrices=expand("data/feature_extraction/{sample}.features",
                                sample=config["samples"])
    output:
        plot="data/pca/pca_1_2.png"
    threads:
        4
    shell:
        "scripts/pca.py --feature-matrices {input.feature_matrices} --plot-fl {output.plot}"
        
# top-level rules
rule setup_inputs:
    input:
        filtered_bams=expand("data/filtered_alignments/{sample}.filtered.bam",
                             sample=config["samples"])

rule count_all_kmers:
    input:
        kmer_counts=expand("data/kmer_counts/{sample}.counts",
                           sample=config["samples"])

