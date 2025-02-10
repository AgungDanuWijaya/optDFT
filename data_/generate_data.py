import mysql.connector
mydb = mysql.connector.connect(
    host="127.0.0.1",
    user="mabok_janda",
    password="yut28092018DAM^",
    database="Quantum",
    auth_plugin='mysql_native_password'
)
mycursor = mydb.cursor()

Q = "SELECT * FROM Quantum.pyscf "
mycursor.execute(Q)
myresult = mycursor.fetchall()
for in_ in range(len(myresult)):

    print(myresult[in_])
    file = open(myresult[in_][1]+"_g", "w")
    file.write(str(myresult[in_][3]))
    file.close()
    file = open(myresult[in_][1] + "_spin", "w")
    file.write(str(myresult[in_][2]))
    file.close()