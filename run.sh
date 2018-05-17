n=0;
max=42;
while [ "$n" -le "$max" ]; do
  mkdir "$n"
  n=`expr "$n" + 1`;
done
