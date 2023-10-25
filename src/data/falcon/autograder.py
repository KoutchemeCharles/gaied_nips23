import __main__ as main

from colorama import Fore, Back, Style

import getpass
import hashlib
import os
import py_compile
import requests
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------
# CS110 Autograder Code (Client i.e., Student Side)
# These functions are used to contact the server
# ---------------------------------------------------------------------

# Constants
autograder_version = 0.42
autograder_ping = "https://autograder.dfcs-cloud.net/versioninfo.php"
autograder_url = "https://autograder.dfcs-cloud.net/api/cs110/get_testcase.php"
scoring_url = "https://autograder.dfcs-cloud.net/api/cs110/upload_result.php"
grades_url = "https://autograder.dfcs-cloud.net/reports.php"
flag_audit = "audit"
flag_show_input = "show_input"
flag_show_output = "show_output"
flag_show_feedback = "show_feedback"
flag_show_score = "show_score"

# Grabs the Contents of the Program that Called It
filename = main.__file__
file = open(filename, "r")
file_path = os.path.abspath(filename)
file_contents = file.read()
debug = False

# Contains a Lookup of Optional Test Case Behavior Flags
# These flags are specified in the unit test under __flags__
custom_flags = {
    flag_audit:False,
    flag_show_input:True,
    flag_show_output:True,
    flag_show_feedback:True,
    flag_show_score:True
    }

# -------------------------------------------------------------
# Prints stylized text to the screen
# usage:  print_styled(Fore.RED, "Red Text Goes here")
# -------------------------------------------------------------
def print_styled(style, text, last_char='\n'):
    #print(style, end='')
    print(text, end='')
    #print(Style.RESET_ALL, end=last_char)


# -------------------------------------------------------------
# Determines if the Device is Connected to the Autograder Service
# -------------------------------------------------------------
def connected_to_internet():
    try:
        _ = requests.get(autograder_ping, timeout=3)
        return True
    except Exception:
        pass

    return False


# -------------------------------------------------------------
# Determines if audit mode has been enabled
# -------------------------------------------------------------
def audit_mode_enabled():
    return custom_flags[flag_audit]


# -------------------------------------------------------------
# Determines if the User would like to run the autograder
# -------------------------------------------------------------
def get_user_preference():
    global connected, debug

    # Automatically Turns off the Autograder if we are Not Connected to the
    # Internet
    if connected:
        if not audit_mode_enabled():
            user_input = input("Test against server? [y/N]: ")
            user_input = user_input.strip().lower()
        else:
            # Automatically sets input to 'n' if audit mode enabled
            user_input = 'n'
    else:
        user_input = 'n'
    print()

    if user_input == 'y' or user_input == 'yes':
        return True
    elif user_input == 'debug':
        debug = True
        return True

    return False


# -------------------------------------------------------------
# Get the current user
# -------------------------------------------------------------
def _get_login():
    username = None
    try:
        username = getpass.getuser()
    except OSError:
        username = os.getlogin()

    return username


# -------------------------------------------------------------
# Extracts Inputs from a List
# -------------------------------------------------------------
def get_inputs(input_list):
    result = ""
    for i in input_list:
        result += str(i) + "\n"
    return result


# -------------------------------------------------------------
# Runs a Python File with the Provided Inputs
# Flags let you specify what outputs, if any, to provide
# -------------------------------------------------------------
def run_script(filename, input_list=[], timeout_in_seconds=5):
        
    # Replaces the Default values with flags (if specified above)
    show_input    = custom_flags[flag_show_input]
    show_output   = custom_flags[flag_show_output]
    show_feedback = custom_flags[flag_show_feedback]
    show_score    = custom_flags[flag_show_score]
    
    # Prints out the Program's Input(s)
    if show_input and len(input_list) > 0:
        print_styled(Style.BRIGHT, "Inputs Provided:")
        for item in input_list:
            print(str(item))
        print()

    # Converts the Input to Bytes
    input_bytes = get_inputs(input_list)

    try:
        # Executes a Subprocess that runs the script with the specified inputs
        p = subprocess.Popen([sys.executable, filename],
                             universal_newlines=True, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=dict(os.environ, DISABLE_AUTOGRADER='1'))
        out, err = p.communicate(input=input_bytes, timeout=timeout_in_seconds)
    except subprocess.TimeoutExpired:
        out = ''
        err = ('Timed out after ' + str(timeout_in_seconds) + ' seconds.  '
               'This can occur when your program asks for more inputs than the '
               'test case provides, or when you have a loop that does not '
               'end.')

    # Prints out the Program's Output
    if show_output:
        print_styled(Style.BRIGHT, "Your Program's Output:")
        if out != '':
            print(out)
        else:
            print("No Output Produced\n")

    # Displays the Error Message (if one is provided)
    if err != '':
        print_styled(Style.BRIGHT + Fore.RED, "Error Occurred:")
        print_styled(Fore.RED, err)
        print()

    # Displays anything that the unit test wants to display
    if show_feedback:
        print_styled(Style.BRIGHT, "Feedback:")

    return (out, err)


# -------------------------------------------------------------
# Reports the Results of the Test, and Removes any Extra Files
# -------------------------------------------------------------
def cleanup(filename, testcase_response_json, points_earned):
    
    # Gets Information Needed from the Testcase Webservice
    timestamp = int(testcase_response_json['timestamp'])
    submissionID = int(testcase_response_json['id'])

    # Sends the Results of the Test Back to the Server for Archiving
    post_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; '
                                  'Intel Mac OS X 10_10_1) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/39.0.2171.95 Safari/537.36'}
    post_data = {
        "user": _get_login(),
        "filename": filename,
        "filepath": file_path,
        "score": points_earned,
        "language": "python",
        "timestamp": timestamp,
        "id":submissionID
    }
    
    response = requests.post(scoring_url, data=post_data, headers=post_headers)
    response_json = response.json()
    response_code = int(response_json['response_code'])
    response_message = str(response_json['message'])

    if response_code == 200:
        print()
        print(response_message)
    else:
        print(response.text)

    # Removes the pycache folder
    if os.path.isdir('__pycache__'):
        shutil.rmtree("__pycache__")


# -------------------------------------------------------------
# Executes the Test Cases
# -------------------------------------------------------------
def run_testcases(test_passed, testcase_response_json, flags={}, perform_cleanup=True):
    # Replaces the Global Custom Flags with the Parameters
    for key in flags:
        custom_flags[key] = flags[key]
    
    # This Flag Determines Whether or Not to Show the Score
    # The value is populated in __flags__
    show_score = custom_flags['show_score']
    
    # Only Runs the Script if Audit Mode is Not Enabled
    if audit_mode_enabled() == False:
        # Runs the Unit Test
        points_earned = test_passed()
        
        # Shows the Score After Everything is Done
        if show_score:
            print_styled(Style.BRIGHT, "\nPoints Earned: " + str(points_earned) + "\n")
        
        # Sends the Results Back to the Autograder Server
        if perform_cleanup:
            cleanup(filename, testcase_response_json, points_earned)
        elif debug:
            print("<SERVER MESSAGE GOES HERE>")
    else:
        if debug:
            print("Audit Mode")
            
        points_earned = 0

    # Returns the Number of Points Earned from the Unit Test
    return points_earned


# -------------------------------------------------------------
# UNIT TEST FUNCTION
# Determines if the Python file provided would run (does not actually run it)
# -------------------------------------------------------------
def code_compiles(filename):
    try:
        py_compile.compile(filename, doraise=True)
        return True
    except Exception:
        return False


# -------------------------------------------------------------
# UNIT TEST FUNCTION
# Determines if two values are numerically equivalent
# -------------------------------------------------------------
def equals(value, expected_value, delta=0.01):
    try:
        return (float(value) >= float(expected_value) - delta and
                float(value) <= float(expected_value) + delta)
    except Exception:
        return value == expected_value


# -------------------------------------------------------------
# UNIT TEST FUNCTION
# Compares two lists of strings to see if they equal
# -------------------------------------------------------------
def compare_strings(student_output_list, expected_output_list, auto_trim=True,
                    check_order=True):
    num_matches = 0

    for i in range(len(student_output_list)):
        print("Line " + str(i+1) + ": ", end='')
        if i < len(expected_output_list):
            if auto_trim:
                student_output_list[i] = student_output_list[i].strip()
                expected_output_list[i] = expected_output_list[i].strip()
            if student_output_list[i] == expected_output_list[i]:
                print("CORRECT")
                num_matches += 1
            else:
                print("INCORRECT (Expected: '{}')".format(expected_output_list[i]))
        else:
            print("INCORRECT (Unexpected Line: '{}')".format(student_output_list[i]))

    print(num_matches, "out of", len(expected_output_list), "lines match")

    return num_matches


# -------------------------------------------------------
# Main Code
# This is the code that a client will execute
# -------------------------------------------------------
def main():
    global connected, debug, custom_flags

    # Checks to See if we are Connected to the Internet
    connected = connected_to_internet()

    # Contacts the API to See if a Test Case Exists
    try:
        print_styled(Style.BRIGHT, "---------------------------------------------------")
        print_styled(Style.BRIGHT, "# Your Program: " + str(filename))
        print_styled(Style.BRIGHT, "---------------------------------------------------")
        
        #if debug or run_autograder:
        if connected:
            print("Autograder (v" + str(autograder_version) + ") Contacting Server . . . ", end='')
            #print("Contacting Test Server . . . ", end='')

            # Generates the Post Data
            post_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; '
                                          'Intel Mac OS X 10_10_1) '
                                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                                          'Chrome/39.0.2171.95 Safari/537.36'}
            post_data = {"user": _get_login(),
                         "filename": filename,
                         "filepath": file_path,
                         "code": file_contents,
                         "language": "python",
                         #"autograde": run_autograder,
                         "autograde": False,
                         "version": autograder_version,
                         }
            
            # Transmits the Code to the Server
            response = requests.post(autograder_url, data=post_data, headers=post_headers)
            response_json = response.json()

            # Extracts the Response Code
            response_code = int(response_json['response_code'])
            
            # Extracts the Message (i.e., the unit test code)
            response_message = response_json['message']

            print("SUCCESS!\n")

            if response_code == 200:
                namespace = globals()
                
                # Compiles the code and stores the values in namespace
                exec(compile(source=response_message, filename='<string>', mode='exec'), namespace)
                
                # Grabs the test_passed() function from the newly compiled code
                test_passed = namespace['test_passed']
                
                # Extracts Optional Flags from the Test Case
                # Replaces Default Flags (Specified at Top of File)
                if '__flags__' in namespace:
                    flags = namespace['__flags__']
                    for key in flags:
                        custom_flags[key] = flags[key]
                                
                # Determines if the User wants to grade the program, or just run it
                # If audit mode is enabled, this will immediately return False
                run_autograder = get_user_preference()
                    
                # Debug
                if debug:
                    print("Debug Information: ")
                    print("API:", response.text)
                    print("Flags:", custom_flags)
                    print()
            
                if run_autograder:
                    # Runs the Test Cases
                    run_testcases(test_passed, response_json)
                    
                    # Prevents Your Program from Running a Second Time
                    sys.exit()
                    
            elif response_code == 505:
                print_styled(Fore.RED, "Autograder Version Not Supported")
                print_styled(Fore.RED, "Update the CS110 module and try again "
                      "(in Thonny, go to Tools -> Manage plug-ins)")
                print("\nRunning Program Locally\n")
                
            else:
                print_styled(Fore.RED + Style.BRIGHT, "Response Code: " + str(response_code))
                print_styled(Fore.RED, "  " + str(response_message))
                print("\nRunning Program Locally\n")
                
        else:
            print_styled(Fore.RED + Style.BRIGHT, "Cannot connect to autograder service.  Wait a bit before trying again.")
            print("\nRunning Program Locally\n")
            
    except Exception as e:
        if connected and (debug):
            print_styled(Fore.RED + Style.BRIGHT, "PROBLEM ENCOUNTERED:")
            print_styled(Fore.RED, e)
            print_styled(Fore.RED, response.text)
            sys.exit()


"""# Prevents a Test Case from Running the Client Main Function
if '_test.py' not in filename and 'DISABLE_AUTOGRADER' not in os.environ:
    main()


"""