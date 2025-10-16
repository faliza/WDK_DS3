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
    type processing cpu
    TO 1
    TO 2
    TO 3
    TO 4

@END

@BEGIN TASK

    ID 1
    type sensing gyroscope accelerometer 
    FROM 0
    TO 5

@END

@BEGIN TASK

    ID 2
    type sensing temperature
    FROM 0
    TO 5

@END

@BEGIN TASK

    ID 3
    type sensing accelerometer
    FROM 0
    TO 6

@END


@BEGIN TASK

    ID 4
    type sensing gyroscope
    FROM 0
    TO 6

@END

@BEGIN TASK

    ID 5
    type processing cpu
    FROM 1
    FROM 2
    TO 7

@END

@BEGIN TASK

    ID 6
    type processing cpu
    FROM 3
    FROM 4
    TO 7

@END

@BEGIN TASK

    ID 7
    type processing cpu
    FROM 5
    FROM 6
    TO 8
    TO 9

@END

@BEGIN TASK

    ID 8
    type processing cpu
    FROM 7

@END

@BEGIN TASK

    ID 9
    type communication zigbee
    FROM 7
    TO 10

@END

@BEGIN TASK

    ID 10
    type communication blu
    TO 9

@END
