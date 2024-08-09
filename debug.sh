NBootstraps=10
SAVERT=/mnt/d/mave_calibration/problematic/
mkdir -p $SAVERT
for bootstrap_iter in $(seq 1 $NBootstraps)
do
    SAVEPATH=$SAVERT/iter_${bootstrap_iter}/
    mkdir -p $SAVEPATH
#     for dataset_id in Erwood_BRCA2_HEK293T \
#     Kato_TP53_AIP1nWT \
#     Kato_TP53_BAXnWT \
#     Kato_TP53_GADD45nWT \
#     Kato_TP53_h1433snWT \
#     Kato_TP53_MDM2nWT \
#     Kato_TP53_NOXAnWT \
#     Kato_TP53_P53R2nWT \
#     Kato_TP53_WAF1nWT
#    do
for dataset_id in Kato_TP53_BAXnWT 
   do
        echo "Running $dataset_id"
        poetry run python main.py \
        --dataset_id=$dataset_id \
        --data_directory=/mnt/i/bio/mave_curation/ \
        --save_path=$SAVEPATH \
        --verbose=False \
        --bootstrap=True >> $SAVEPATH/$dataset_id.log 2>&1;
    done
done