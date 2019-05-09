dir_name='2015CS10435'

rm -rf $dir_name
rm -rf ${dir_name}.zip
mkdir $dir_name
cp train.py $dir_name/
cp test.py $dir_name/
cp utils.py $dir_name/
cp run.sh $dir_name/
cp compile.sh $dir_name/
cp writeup.txt $dir_name/
cp requirements.txt $dir_name/

zip -r ${dir_name}.zip $dir_name
rm -rf $dir_name