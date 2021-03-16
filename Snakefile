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
        chrom=config["chrom"],
        kmer_size=config["kmer_size"]
    output:
        jf="data/kmer_counts/{sample}.jf"
    threads:
        4
    shell:
        "samtools view -b -@{threads} -F 0x0200 -F 0x0100 -F 0x004 -q {params.min_mapping_qual} {input.bam} {params.chrom} | samtools fasta - |  jellyfish count -t {threads} -m {params.kmer_size} -s 1000M -C -o {output.jf} /dev/fd/0"


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
        "jellyfish dump -c -L 2 {input.jf} > {output.counts}"

rule extract_features:
    input:
        kmer_counts="data/kmer_counts/{sample}.counts",
    params:
        n_features=config["n_features"],
        use_binary=config["use_binary_features"]
    output:
        feature_matrix="data/feature_extraction/{sample}.features"
    threads:
        2
    shell:
        "scripts/feature_extractor.py --kmer-freq-fl {input.kmer_counts} --n-features {params.n_features} {params.use_binary} --feature-matrix {output.feature_matrix}"

rule log1p_feature:
    input:
        kmer_counts="data/kmer_counts/{sample}.counts",
    params:
         n_features=config["n_features"]
    output:	
         feature_matrix="data/log1p_features/{sample}.features"
    threads:
         4
    shell:
        "scripts/feature_extractor.py --kmer-freq-fl {input.kmer_counts} --n-features {params.n_features} --feature-matrix {output.feature_matrix} --counts --log1p"	

rule counts_feature:
    input:
        kmer_counts="data/kmer_counts/{sample}.counts",
    params:
        n_features=config["n_features"]
    output:
        feature_matrix="data/count_features/{sample}.features"
    threads:
        2
    shell:
        "scripts/feature_extractor.py --kmer-freq-fl {input.kmer_counts} --n-features {params.n_features} --feature-matrix {output.feature_matrix} --counts"

rule binarizer_feature:
    input:
        kmer_counts="data/kmer_counts/{sample}.counts",
    params:
        n_features=config["n_features"]
    output:
        feature_matrix="data/binarizer_features/{sample}.features"
    threads:
        2
    shell:
        "scripts/feature_extractor.py --kmer-freq-fl {input.kmer_counts} --n-features {params.n_features} --feature-matrix {output.feature_matrix} --counts --binarizer"

rule log1p_no_count:
    input:
        kmer_counts="data/kmer_counts/{sample}.counts",
    params:
        n_features=config["n_features"]
    output:
        feature_matrix="data/log1p_no_counts/{sample}.features"
    shell:
        "scripts/feature_extractor.py --kmer-freq-fl {input.kmer_counts} --n-features {params.n_features} --feature-matrix {output.feature_matrix} --binary --log1p"


rule binarizer_no_count:
    input:
        kmer_counts="data/kmer_counts/{sample}.counts",
    params:
        n_features=config["n_features"]
    output:
        feature_matrix="data/binarizer_no_counts/{sample}.features"
    shell:
        "scripts/feature_extractor.py --kmer-freq-fl {input.kmer_counts} --n-features {params.n_features} --feature-matrix {output.feature_matrix} --binary  --binarizer"


rule binary_features:
    input:
        kmer_counts="data/kmer_counts/{sample}.counts",
    params:
        n_features=config["n_features"]
    output:
        feature_matrix="data/binary_features/{sample}.features"
    shell:
        "scripts/feature_extractor.py --kmer-freq-fl {input.kmer_counts} --n-features {params.n_features} --feature-matrix {output.feature_matrix} --binary"

#This is pca rules for binary/no binary
rule pca_binary:
    input:
        feature_matrices=expand("data/binary_features/{sample}.features",
                               sample=config["samples"])
    params:
        groups_fl=config["groups_fl"]
    output:
        plot="data/pca/pca_binary.png"
    threads:
        2
    shell:
        "scripts/pca.py --feature-matrices {input.feature_matrices} --groups-fl {params.groups_fl} --plot-fl {output.plot}"

rule pca:
    input:
        feature_matrices=expand("data/feature_extraction/{sample}.features",
                                sample=config["samples"])
    params:
        groups_fl=config["groups_fl"]
    output:
        plot="data/pca/pca_1_2.png"
    threads:
        2
    shell:
        "scripts/pca.py --feature-matrices {input.feature_matrices} --groups-fl {params.groups_fl} --plot-fl {output.plot}"

rule pca_log1p:
    input:
       feature_matrices=expand("data/log1p_features/{sample}.features", 
			                        	sample=config["samples"])
    params:
       groups_fl=config["groups_fl"]
    output:
       plot="data/pca/pca_log1p.png"
    threads:
       2
    shell:
        "scripts/pca.py --feature-matrices {input.feature_matrices} --groups-fl {params.groups_fl} --plot-fl {output.plot}"

rule pca_binarizer:
    input:
        feature_matrices=expand("data/binarizer_features/{sample}.features",
                                 sample=config["samples"])
    params:
        groups_fl=config["groups_fl"]
    output:
        plot="data/pca/pca_binarizer.png"
    threads:
        2
    shell:
        "scripts/pca.py --feature-matrices {input.feature_matrices} --groups-fl {params.groups_fl} --plot-fl {output.plot}"

rule pca_log1p_no_count:
    input:
       feature_matrices=expand("data/log1p_no_counts/{sample}.features", 
				                        sample=config["samples"])
    params:
       groups_fl=config["groups_fl"]
    output:
       plot="data/pca/pca_log1p_no_count.png"
    threads:
       2
    shell:
        "scripts/pca.py --feature-matrices {input.feature_matrices} --groups-fl {params.groups_fl} --plot-fl {output.plot}"

rule pca_binarizer_no_count:
    input:
        feature_matrices=expand("data/binarizer_no_counts/{sample}.features",
                                 sample=config["samples"])
    params:
        groups_fl=config["groups_fl"]
    output:
        plot="data/pca/pca_binarizer_no_counts.png"
    threads:
        2
    shell:
        "scripts/pca.py --feature-matrices {input.feature_matrices} --groups-fl {params.groups_fl} --plot-fl {output.plot}"


# top-level rules
rule setup_inputs:
    input:
        filtered_bams=expand("data/filtered_alignments/{sample}.filtered.bam",
                             sample=config["samples"])

rule count_all_kmers:
    input:
        kmer_counts=expand("data/kmer_counts/{sample}.counts",
                           sample=config["samples"])

