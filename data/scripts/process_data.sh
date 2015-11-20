IN=$1
OUT=$2
TMP=$OUT/tmp/
N=200
SPLITTA=/Users/quique/code/python/splitta/

mkdir $TMP

echo "text-processing sentences files"
for f in ${IN}*; do
    python $SPLITTA/sbd.py -m $SPLITTA/model_svm $f \
	| python ./text_preprocessing.py > $OUT`basename $f`
done

echo "cleaning up"
rm -r $TMP

echo "done!"
