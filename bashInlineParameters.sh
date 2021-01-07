#!/bin/bash

mainDirectory="./" #Edit this if you are not running everything from the corresponding file
logDirectory=$mainDirectory"/output" #Just a placeholder name for where you want your output
editFile="fullSetupLosslessBinarizationTsetlin.py"
cd $mainDirectory

function runSpecificSettings() {
	helpString="Images-"$1"_Epochs-"$2"_Type-"$3"_Method-"$4"_T-"$5"_S-"$6"_Clauses-"$7
	#help2="date +%D_%T"
	help2=""
	helpString=$helpString$help2
	typeHelp=""
    methodHelp=""
    if [[ ${3:0:1} == "p" ]]; then #Just a test of a substring method as opposed to the wildcard regex examples below
        typeHelp="paralell"
    elif [[ $3 == c* ]]; then #Wildcard regex for the first letter being c and then whatever
        typeHelp="cuda"
    else
        echo "No valid input for type try paralell or cuda"
    fi
	if [[ $4 == l* ]]; then
        methodHelp=1
    elif [[ $4 == g* ]]; then
        methodHelp=0
    else
        echo "No valid input for method try lossless or gauss"
    fi
	##For loop to create cats
	sed -i "22s/.*/imagesToTrainOn = "$1"/" $editFile
	sed -i "23s/.*/epochs = "$2"/" $editFile
	sed -i "19s/.*/tsetlinType = tsetlinTypes[\""$typeHelp"\"]/" $editFile
	sed -i "32s/.*/informationProcessingType = "$methodHelp"/" $editFile
	sed -i "26s/.*/T = "$5"/" $editFile
    sed -i "27s/.*/s = "$6".0/" $editFile
    sed -i "25s/.*/clauses = "$7"/" $editFile
	#touch $logDirectory/"results_"$helpString
	python $editFile >> $logDirectory/"results_"$helpString
    #python $editFile >> $logDirectory/"results_"$helpString &
	#echo $1$2$3$4$5

}
if [[ $1 == a* ]]; then
    #runSpecificShit 50000 300 para lossless 75 10 3200
    runSpecificSettings 50000 300 para gauss 75 10 800
elif [ -z $1 ] || [ -z $2 ] || [ -z $3 ] || [ -z $4 ] || [ -z $5 ] || [ -z $6 ] || [ -z $7 ]; then
	echo "Needs More arguments"
	echo "Images, Epochs, para/cuda, gauss/lossless, T, s, Clauses"
    echo "ex: \"500, 50, para, lossless, 75, 10, 800\""
else
	#Categories, T, s, clauses, states
	runSpecificSettings $1 $2 $3 $4 $5 $6 $7
	#echo "do i end up here?"
fi