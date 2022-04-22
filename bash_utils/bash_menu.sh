#!/bin/bash

# Bash Menu Script
files=( ${1}/*${2} )

## Enable extended globbing. This lets us use @(foo|bar) to
## match either 'foo' or 'bar'.
shopt -s extglob

## Start building the string to match against.
string="@(${files[0]}"
## Add the rest of the files to the string
for((i=1;i<${#files[@]};i++))
do
    string+="|${files[$i]}"
done
## Close the parenthesis. $string is now @(file1|file2|...|fileN)
string+=")"

## Show the menu. This will list all files and the string "quit"
# printf "Select path to csv file for training the model\n"
select FILEPATH in "${files[@]}" "manual path" "quit"
do
    case $FILEPATH in
    ## If the choice is one of the files (if it matches $string)
    $string)
        ## Do something here
        break;
        ;;

    "manual path")
        # read in user input
        echo "Give path to file"
        read -e FILEPATH
        break;
        ;;

    "quit")
        ## Exit
        exit;;
    *)
        file=""
        echo "Please choose a number from 1 to $((${#files[@]}+2))";;
    esac
done

# printf "CSV file $FILEPATH will be used for modelling\n"
