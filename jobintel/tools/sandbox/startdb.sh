#!/bin/sh
# idempotent: start the MariaDB test server only if it is not already answering
if mariadb -ujobintel -ptestpw -h127.0.0.1 -P3307 -e "select 1" >/dev/null 2>&1; then exit 0; fi
mkdir -p /run/mysqld && chown mysql:mysql /run/mysqld
setsid nohup mariadbd --user=mysql --datadir=/tmp/mysqldata --socket=/tmp/mysqlrun/mysql.sock --pid-file=/tmp/mysqlrun/mysqld.pid --port=3307 --bind-address=127.0.0.1 --innodb-buffer-pool-size=256M --max-allowed-packet=256M --innodb-use-native-aio=0 >/tmp/mdb.log 2>&1 </dev/null &
for i in $(seq 1 40); do sleep 1; mariadb -ujobintel -ptestpw -h127.0.0.1 -P3307 -e "select 1" >/dev/null 2>&1 && exit 0; done
echo "DB FAILED TO START"; tail -5 /tmp/mdb.log; exit 1
