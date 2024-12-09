#!/usr/bin/env python3
import os
import sys
import glob
import time
import argparse
import subprocess
# from utilities import ongoing_run_time

def main():
    
    # Operation choices
    choices = ['cut', 'rename', 'sample', 'filter', 'edit']

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Python wrapper for batch grid operations using GMT.')

    # Positional arguments
    # parser.add_argument('input_file', type=str, help='Path to the input file')

    # Optional arguments
    # parser.add_argument('-o', '--output', type=str, help='Path to the output file (required for "process" operation)')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')

    # operation argument (specifies which function to call)

    # ------------------ General ------------------
    parser.add_argument('--operation', type=str, choices=choices, required=True,
                        help='operation to perform: "process" or "analyze"')

    parser.add_argument('--check', type=bool, default=False,
                        help='Check batch commands without running operation')



    # ------------------ rename ------------------
    parser.add_argument('--search', type=str,
                        help='Wildcard string for getting file list."')

    parser.add_argument('--delete', nargs='+', type=int, default=[0, 0],
                        help='Delete characters from start or end of file name. Format: start end')

    parser.add_argument('--append', type=str, default='',
                        help='Append characters to end of file name.')

    parser.add_argument('--prepend', type=str, default='',
                        help='Prepend characters to start of file name.')

    # ------------------ cut ------------------
    # parser.add_argument('--region', type=str, default='-180/180/-90/90',
    #                     help='Region for cutting grids. Format: xmin xmax ymin ymax')
    parser.add_argument('--region', nargs='+', type=str, default=['-180', '180', '-90', '90'],
                    help='Region for cutting grids. Format: xmin xmax ymin ymax')

    # ------------------ sample ------------------
    parser.add_argument('--inc', nargs='+', type=str, default=['1', '1', ],
                help='x/y increments for resampling grids. Format: xinc yinc')

    parser.add_argument('--target_grid',  type=str, default='',
                        help='Path to target grid for resampling/cutting')

    # ------------------ sample ------------------
    parser.add_argument('--dist_code',  type=str, default='1',
                        help='Distance code: call "gmt grdfilter" for options')
    parser.add_argument('--filter',  type=str, default='g20+h',
                    help='Filter code: call "gmt grdfilter" for options')

    # ------------------ Edit ------------------
    parser.add_argument('--options', nargs='+', type=str,
                        help='Specify "gmt grdedit" options. NOTE: omit "-" before flags')

    # Parse arguments
    args = parser.parse_args()

    # Route to the appropriate function based on the operation
    if args.operation == 'rename':
        if not args.search:
            print('Error: The "rename" operation requires the --search argument.')
            sys.exit(1)

        rename(args.search, args.delete, args.prepend, args.append, args.check)

    elif args.operation == 'cut':
        if (not args.search) | (not args.region):
            print('Error: The "cut" operation requires the --search and --region arguments.')
            sys.exit(1)

        cut(args.search, args.region, args.delete, args.prepend, args.append, args.check)

    elif args.operation == 'sample':
        if (not args.search) | ((not args.inc) & (not args.target_grid)):
            print('Error: The "cut" operation requires the --search and --region arguments.')
            sys.exit(1)

        sample(args.search, args.inc, args.target_grid, args.delete, args.prepend, args.append, args.check)

    elif args.operation == 'filter':
        if not args.search:
            print('Error: The "filter" operation requires the --search argument')
            sys.exit(1)

        filter(args.search, args.dist_code, args.filter, args.delete, args.prepend, args.append, args.check)

    elif args.operation == 'edit':
        if not args.search:
            print('Error: The "edit" operation requires the --search argument')
            sys.exit(1)

        edit(args.search, args.options, args.check)
    else:
        print(f'Error: must specify operation from {choices}')
        sys.exit(1)



# -------------------------- Operation drivers --------------------------

def edit(search, options, check):
    """
    Use grdedit to modify grid info
    """

    # Get paths and filenames
    paths, files = get_paths(search)

    # Perform editing
    for path, file in zip(paths, files):

        cmd = ['gmt', 'grdedit', f'{path}{file}'] + [f'-{option}' for option in options]
        if check:
            print(' '.join(cmd))
        else:
            subprocess.run(cmd) 
    return


def filter(search, dist_code, filter, delete, prepend, append, check):
    """
    Filter grids using grdfilter 
    Default suffix for cut files is  '_filt' unless specified otherwise.
    """ 

    if append.lower() == 'none':
        suffix = ''
    elif len(append) > 0:
        suffix = append
    else:
        suffix = '_filt'

    # Get paths and filenames
    paths, files = get_paths(search)

    start = time.time()
    time_prev = time.time()
    
    
    # Perform filtering
    for path, file in zip(paths, files):

        # Set file name
        new_file = prepend + file[delete[0]:-4 - delete[1]] + suffix + '.grd'

        # Get command
        cmd = ['gmt', 'grdfilter', f'{path}{file}', f'-D{dist_code}', f'-F{filter}', f'-G{path}{new_file}']
   
        if check:
            print(' '.join(cmd))
        else:
            subprocess.run(cmd) 

            ongoing_run_time(f'{new_file} finished: ', time_prev)
            time_prev = time.time()


def sample(search, inc, target_grid, delete, prepend, append, check):
    """
    Resample set of files based on specified x/y increments or target grid file. 
    Default suffix for cut files is  '_samp' unless specified otherwise.
    """ 

    if append.lower() == 'none':
        suffix = ''
    elif len(append) > 0:
        suffix = append
    else:
        suffix = '_samp'

    # Get paths and filenames
    paths, files = get_paths(search)

    # Perform renaming
    for path, file in zip(paths, files):

        # Set cut file name
        new_file = prepend + file[delete[0]:-4 - delete[1]] + suffix + '.grd'

        # Get command
        if len(target_grid) > 0:
            cmd = ['gmt', 'grdsample', f'{path}{file}', f'-R{target_grid}', f'-G{path}{new_file}']
        else:
            cmd = ['gmt', 'grdsample', f'{path}{file}', f'-I{"/".join(inc)}', f'-G{path}{new_file}']

        if check:
            print(' '.join(cmd))
        else:
            subprocess.run(cmd) 


def cut(search, region, delete, prepend, append, check):
    """
    Cut set of files based on specified region. 
    Default suffix for cut files is  '_cut' unless specified otherwise.
    """ 

    if append.lower() == 'none':
        suffix = ''
    elif len(append) > 0:
        suffix = append
    else:
        suffix = '_cut'


    # Get paths and filenames
    paths, files = get_paths(search)

    # Perform renaming
    for path, file in zip(paths, files):

        # Set cut file name
        # cut_file = file[:-4] + suffix + '.grd'
        cut_file = prepend + file[delete[0]:-4 - delete[1]] + suffix + '.grd'

        # Get command
        cmd = ['gmt', 'grdcut', f'{path}{file}', f'-R{"/".join(region)}', f'-G{path}{cut_file}']

        if check:
            print(' '.join(cmd))
        else:
            subprocess.run(cmd) 

    sys.exit(1)


def rename(search, delete, prepend, append, check):
    """
    Rename set of files based on specified format. 
    """

    # Get paths and filenames
    paths, files = get_paths(search)

    # Perform renaming
    for path, file in zip(paths, files):

        # Get new name
        new_file = prepend + file[delete[0]:-4 - delete[1]] + append + '.grd'

        if len(path) == 0:
            old_path = file
            new_path = new_file
        else:
            old_path = path + '/' + file
            new_path = path + '/' + new_file

        # Rename
        if check:
            print('Renaming', old_path, 'to', new_path)
        else:
            os.rename(old_path, new_path)

    sys.exit(1)


# -------------------------- Utilities --------------------------
def get_paths(search):
    """
    Use glob to get list of files and split into filenames and directory paths.
    """

    # Get complete paths
    full_paths = sorted(glob.glob(search))

    # Get file names
    files = [path.split('/')[-1] for path in full_paths]

    # Get directory paths, accounting for files being in the current directory
    paths = []
    for path in full_paths:
        split_path = path.split('/')[:-1]
        if len(split_path) > 0:
            paths.append('/'.join(split_path) + '/')
        else:
            paths.append('')

    return paths, files


def get_full_path(path, file):
    """
    Helper method to reconstruct full path
    """

    if len(path) == 0:
        full_path = file
    else:
        full_path = path + '/' + new_file
    return full_path


def process_input(input_file, output_file, verbose=False):
    """
    Process the input file and write to output file.
    """
    if verbose:
        print(f'Processing input file: {input_file} and writing to {output_file}')

    try:
        with open(input_file, 'r') as infile:
            data = infile.read()
        
        with open(output_file, 'w') as outfile:
            outfile.write(data)

        if verbose:
            print(f'Data from {input_file} has been written to {output_file}.')

    except FileNotFoundError:
        print(f'Error: File {input_file} not found.')
        sys.exit(1)
    except Exception as e:
        print(f'An error occurred: {e}')
        sys.exit(1)


def ongoing_run_time(message, time_prev, units='s'):
    """
    Print operation run time and return current time.
    Time is printed at end of message with units appended.
    Default is seconds ('s')
    """

    time_now = time.time()
    dt       = time_now - time_prev

    if units != 's':
        if units == 'min':
            dt /= 60
        elif units == 'hr':
            dt /= 3600 

    print(f'{message} {dt:.1f} {units}')

    return time_now

if __name__ == '__main__':
    main()