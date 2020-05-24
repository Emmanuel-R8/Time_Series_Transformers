# python gpu_perf_vs_shape.py -l 1 2 4 6 9 12 16 20 25 32 -d 8 16 24 32 48 64 96 128 192 256 384 -b 1 2 5 10 20 40 60 100 --fp16 O0
for l in 1 2 4 6 9 12 16 20 25 32
do
    for d in 8 16 32 64 128 256 512
    do
        for b in 1 2 4 8 16 32 64 128
        do
            for amp_mode in O1 O2 O0
            do
                python -W ignore gpu_perf_vs_shape.py -l $l -d $d -b $b --fp16 $amp_mode --reload --tracking
            done
        done
    done
done
