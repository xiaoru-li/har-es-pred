# This is a script to format the raw data from the training logs into a CSV file

# The input file is a text file with a lot of extra information we don't need. We only need the rows starting with `Epoch:  `.
# The useful rows are formatted like this:
# `Epoch:  899  Training Loss:  0.11956885029063667  Training Accuracy:  0.9635663786331501  f1:  0.4974051181461314  Validation Accuracy:  0.8692834322227293`
# We need to extract the epoch number, training loss, training accuracy, f1 score, and validation accuracy, and put them in a CSV file.

# The output file is a CSV file with the following columns:
# epoch,t_loss,t_acc,f1,v_acc

# import the necessary libraries
import csv
import re
import sys

# open the input file, the file name is passed as a command line argument
with open(sys.argv[1], 'r') as f:
    # open the output file, which is the same name as the input file, but with a .csv extension
    with open(sys.argv[1].split('.')[0] + '.csv', 'w+') as g:
        # create a CSV writer object
        writer = csv.writer(g)
        # write the column names
        writer.writerow(['epoch', 't_loss', 't_acc', 'f1', 'v_acc'])
        # loop over each line in the input file
        for line in f:
            # if the line starts with `Epoch:  ` and contains the words `Training Loss:  `, `Training Accuracy:  `, `f1:  `, and `Validation Accuracy:  `
            if line.startswith('Epoch:  ') and 'Training Loss:  ' in line and 'Training Accuracy:  ' in line and 'f1:  ' in line and 'Validation Accuracy:  ' in line:
                # extract the epoch number
                epoch = re.search('Epoch:  (\d+)', line).group(1)
                # extract the training loss
                t_loss = re.search('Training Loss:  ([\d\.]+)', line).group(1)
                # extract the training accuracy
                t_acc = re.search('Training Accuracy:  ([\d\.]+)', line).group(1)
                # extract the f1 score
                f1 = re.search('f1:  ([\d\.]+)', line).group(1)
                # extract the validation accuracy
                v_acc = re.search('Validation Accuracy:  ([\d\.]+)', line).group(1)
                # write the data to the output file
                writer.writerow([epoch, t_loss, t_acc, f1, v_acc])

# close the files
f.close()
g.close()
