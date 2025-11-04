# This is a comment
# A sample application specification file
# The fields read here should be parsed to the target Application
# FORMAT: 
#   - Each new task begins with "@BEGIN TASK", and ends with "@END"
#   - The keyword "ID" of the corresponding task
#   - The keyword "from" specifies the IDs of the tasks that lead to this task (the in-coming arcs)
#   - The keyword "to" specifies the IDs of the tasks this task connects to (the out-going arcs)
#   - The keyword "type" specifies the "type of the task". It can be used as follows:
#       . type sensing gyroscope ("sensing" specifies the nature of the task, gyroscope gives the precise functionality)
#       . type processing cpu (see the README or the help file for the complete description)
#       . type communication ble 

@BEGIN TASK

    ID 0
    type sensing gyroscope
    TO 1

@END

@BEGIN TASK

    ID 1
    type processing cpu
    TO 2

@END

@BEGIN TASK

    ID 2
    type processing cpu 
    TO 3

@END

@BEGIN TASK

    ID 3
    type processing cpu
    TO 4

@END

@BEGIN TASK

    ID 4
    type processing cpu
    TO 5

@END
