import sqlite3

conn = sqlite3.connect('f1.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM dim_driver')
print('drivers:', cursor.fetchone()[0])

cursor.execute('SELECT COUNT(*) FROM dim_team')
print('teams:', cursor.fetchone()[0])

cursor.execute('SELECT COUNT(*) FROM dim_tyre_compound')
print('compounds:', cursor.fetchone()[0])

cursor.execute('SELECT * FROM dim_driver LIMIT 5')
print(cursor.fetchall())

conn.close()