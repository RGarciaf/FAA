if [ $# -eq 1 ]; then
	git add -A
	git commit -m $1
	git push
elif [ $# -eq 0 ]; then
	git add -A
	git commit
	git push
fi
