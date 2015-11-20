IN=$1
OUT=$2
TMP=$OUT/tmp/
N=200
SPLITTA=/home/enrique/code/splitta.1.03

mkdir $TMP

echo "text-processing sentences files"
for f in ${IN}*; do
    python $SPLITTA/sbd.py -m $SPLITTA/model_svm $f \
	| python ./text_preprocessing.py > $OUT`basename $f`
done

echo "cleaning up"
rm -r $TMP

echo "done!"
