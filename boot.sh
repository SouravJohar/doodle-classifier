kill -9 $(lsof -t -i:5000)
python -W ignore server.py &
sleep 5
npm start
kill -9 $(lsof -t -i:5000)
