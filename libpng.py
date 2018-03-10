from subprocess import call
import os

for i in os.listdir("/home/victor/Projects/INRIAPerson/Train/neg/"):
    call(["pngcrush", "-fix", "-force", "/home/victor/Projects/INRIAPerson/Train/neg/" + i,
          "/home/victor/Projects/INRIAPerson/NEWTRAINING/libpngNEG/" + i])
