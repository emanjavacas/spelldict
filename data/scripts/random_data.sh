N=200
IN=/Users/quique/corpora/EEBO/EEBO_chadwyck/
RAW=/Users/quique/code/python/spelldict/data/raw/
SBD=/Users/quique/code/python/spelldict/data/post/

echo "processing input files"
ls $IN | gsort -R | tail -$N | while read f; do
    python /Users/quique/code/MBG/formatting/chadwyck/EEBO_raw_text.py $IN$f `# extract raw text` \
    | tr -d '' `# remove newline carriage chars` \
    | sed '/^[ \t]*$/d' `# remove multiple newline chars` \
    > $RAW$f
done

echo "segmenting sentences files"
for f in ${RAW}*; do
    python /Users/quique/code/python/splitta.1.03/sbd.py -m /Users/quique/code/python/splitta.1.03/model_svm $f > $SBD`basename $f`
done
