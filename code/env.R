# Environment script
# This must be run before anything else

rm(list=ls())
gc()
cat("\014")

# Load required packages

library(tidyverse)
library(yaml)
library(stargazer)
library(xtable)

# Set working directory to current location and define relative file paths
setwd(dirname(sys.frame(1)$ofile))
setwd("..")
input_dir <- file.path(getwd(),"data","input")
output_dir <- file.path(getwd(),"data","output")
code_dir <- file.path(getwd(),"code")
config_dir <- file.path(code_dir,"config")
script_dir <- file.path(code_dir,"scripts")
utilities_dir <- file.path(code_dir,"utilities")
tex_dir <- file.path(getwd(),"docs")

# Call all user-defined functions in the utility directory
# file_list <- list.files(path = utilities_dir,pattern="*.R")
# invisible(sapply(paste(utilities_dir,file_list,sep = "/"),source,.GlobalEnv))
# rm(file_list)

# Load config file

config <- read_yaml(file.path(config_dir,"config.yaml"))
