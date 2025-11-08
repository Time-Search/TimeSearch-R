ps aux|grep $1|grep -v grep|cut -c 9-16|xargs kill -9
