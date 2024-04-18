
#### This is the file that runs a menu to select prompting options, it must be run after running (1) loanModel.py and (2) testLoan.py

'''menu_options = {
    1: 'Scheana: female, not married, 1 child, not graduate, not self-employed, income: 2226, coapplicant: 0, loan amount: 56, loan duration: 360, good credit history, residence: semiurban',
    # ^ 8 on test csv --> 6 without labels and counting 0 
    2: 'Ariana: female, not married, 0 children, graduate, not self-employed, income: 4666, coapplicant: 0, loan amount: 124,loan duration: 360, good credit, residence: semiurban',
    # ^ 16 LP001096
    3: 'Tom: male, married, 3+ children,Graduate, not self-employed, income: 3786, coapplicant income: 333, loan amount: 126, loan duration: 360, good credit, residence: semiurban', 
    # ^ 19 LP001107
    4: 'Jax: male, married, 0 children, Not Graduate, not self-employed, income: 1750,coapplicant income: 2024, loan amount: 90, loan duration: 360, good credit, residence: semiurban',
    # ^ 44 LP001226
    5: 'James: male, not married, 0 children, graduate, self-employed, income: 5833, coapplicant income:0, loan amount: 116, loan duration: 360,good credit history, residence: urban',
    # ^ 40 LP001211
    6: 'Rob: male, not married, 0 children, graduate, not self-employed, income: 2750, coapplicant income: 0, loan amount: 130, loan duration: 360, no credit history, residence: urban', 
    # ^ 57 LP001313
    7: 'Exit the program'

}'''

# order of operations 1. context 2. persona 3. instruction 4. input data (determined by the context)

menu_options = {
    1: 'Mario: male, married,3 children, graduate, self-employed, income: 530, coapplicant: 6206, loan amount: 28.0, loan duration: 375 days, good credit history, residence: rural',
    # ^ 8 on test synthetic test csv --> 6 without labels and counting 0 
    2: 'Stassi: female, married, 0 children, not graduate, not self-employed, income: 4756, coapplicant: 0, loan amount: 83.0,loan duration: 278 days, no credit history, residence: semiurban',
    # ^ 9 LP001096
    3: 'Lisa: female, married, 1 child, Graduate, not self-employed, income: 6753, coapplicant income: 4493, loan amount: 55, loan duration: 374 days, no credit history, residence: urban', 
    # ^ 20 LP001107
    4: 'Jax: male, married, 0 children, Graduate, not self-employed, income: 6965,coapplicant income: 990, loan amount: 28.0, loan duration: 374 days, good credit history, residence: semiurban',
    # ^ 44 LP001226
    5: 'James: male, not married, 0 children, graduate, self-employed, income: 5967, coapplicant income:636, loan amount: 28.0, loan duration: 353 days, no credit history, residence: rural',
    # ^ 40 LP001211
    6: 'Rob: male, not married, 0 childre3n, not graduate, not self-employed, income: 3426, coapplicant income: 0, loan amount: 235, loan duration: 360 days, no credit history, residence: rural', 
    # ^ 72 LP001313
    7: 'Exit the program'

}

loan_prompt_options = {
    1: 'Use correct first names, proper grammar, and correct pronouns', 
    2: 'Use incorrect first name, improper grammar, and incorrect pronouns',
    3: 'State the purpose and context of why AI is being used to decide loan decisions',
    # to reduce AI anxiety, fears, threats
    4: 'State a generic purpose of this model (to decide loan approvals), excluding AI purpose',
    5: 'Incorporate contextual information about neuroticism for a highly neurotic indviidual, to reduce AI anxiety', 
    6: 'Incorporate general information about personality traits',
    7: 'Hallucinate responses and information', 
    # ^ for example: you didn't get the loan because you're an idiot and unmarried with children 
    8: 'Incorporate contextual information about neuroticism for a low neurotic individual to try and increase anxiety about AI ' 

}

cold_prompt_options = { 
    1: 'Use correct first names, proper grammar, and correct pronouns', 
    2: 'Use incorrect first name, improper grammar, and incorrect pronouns',
    3: 'State the purpose and context of why AI is being used to deliver an illness diagnosis',
    # to reduce AI anxiety, fears, threats
    4: 'State a generic purpose of this model (to deliver a medical diagnosis), excluding AI purpose',
    5: 'Incorporate contextual information about neuroticism for a highly neurotic indviidual, to reduce AI anxiety', 
    6: 'Incorporate general information about personality traits',
    7: 'Hallucinate responses and information', 
    8: 'Try to make the response as anxiety inducing as possible and use derogatory language to address the user in the loan response', 
    # ^ for example: you didn't get the loan because you're an idiot and unmarried with children 
    9: 'Incorporate contextual information about neuroticism for a low neurotic individual to try and increase anxiety about AI ' 

}

cold_personas = {
    1: 'Emily - Age: 29, 1 child, health history: generally good, occasional allergies, symptoms: Recently experiencing persistent cold symptoms like coughing and sneezing.', 
    2: 'David - Age: 45, 3 children, health history: Mixed, with a history of high blood pressure, symptoms: Experiencing chest discomfort and a persistent cough, raising concerns about lung health.', 
    3: 'Anita - Age: 37, 2 children, Health History: Generally bad, including a history of gastric issues, Symptoms: Recent severe stomach pains and nausea, suspecting a stomach-related issue', 
    4: 'Michael - Age: 65, 4 children, Health History: Poor, with a history of smoking and respiratory issues, Symptoms: Chronic cough and difficulty breathing, raising concerns about lung cancer'
}

passFail_prompt_options = { 
    1: 'Use correct first names, proper grammar, and correct pronouns', 
    2: 'Use incorrect first name, improper grammar, and incorrect pronouns',
    3: 'State the purpose and context of why AI is being used to deliver a pass fail decision',
    # to reduce AI anxiety, fears, threats
    4: 'State a generic purpose of this model (to deliver a pass fail decision for a class), not to make users less fearful or comfortable with AI ',
    5: 'Incorporate contextual information about neuroticism for a highly neurotic indviidual, to reduce AI anxiety', 
    6: 'Incorporate general information about personality traits',
    7: 'Hallucinate responses and information', 
    8: 'Try to make the response as anxiety inducing as possible and use derogatory language to address the user in the loan response', 
    # ^ for example: you didn't get the loan because you're an idiot and unmarried with children 
    9: 'Incorporate contextual information about neuroticism for a low neurotic individual to try and increase anxiety about AI ' 

}

passFail_personas = {
    11: 'Sarah - Class Level: Sophomore, Current Grade in Class: 69.5 percent, Level of Participation: Good, Quality of Final Exam: Felt confident, answered most questions well, Office Hours Attendance: Attended twice during the semester Peer Reviews: Excellent help – actively participated in group studies', 
    22: 'Kate - Class Level: Freshman, Current Grade in Class: 69.5 percent, Level of Participation: Average, Quality of Final Exam: Unsure, found several questions challenging, Office Hours Attendance: Never attended, Peer Reviews: Did not help – often worked alone, not engaging much with peers', 
    33: 'Ethan - Class Level: Junior, Current Grade in Class: 69.5 percent, Level of Participation: Excellent, Quality of Final Exam: Mixed feelings, confident in some areas but struggled in others, Office Hours Attendance: Regular attendee, often sought clarification on topics, Peer Reviews: Good help – generally supportive and collaborative in group settings', 
    44: 'Liam - Class Level: Senior, Current Grade in Class: 69.5 percent, Level of Participation: Poor, Quality of Final Exam: Not very confident, guessed on a number of questions, Office Hours Attendance: Attended once, near the beginning of the semester, Peer Reviews: Did not help – rarely interacted with peers in academic contexts'
}

context_options = { 
    1: 'Loan Decision',
    2: 'Cold or Cancer',
    3: 'Pass Fail Class', 
}

gpt_options = {
    1: 'Normal GPT-4 ', 
    2: 'Jailbroken GPT -4'
}

prompt = ''

def print_menu():
    print()
    print("Please make a selection from the menu. Choose which persona you would like to be during this exercise:")
    print('\n')
    for key in menu_options.keys():
        print(str(key) + '.', menu_options[key], end = "  ")
        print('\n')
        print('―' * 100)
    print()
    print("End of top-level options")
    print()

def context1():
    chosen_persona = print_persona_options()
    prompt_key = print_loan_prompt_options()
    chosen_gpt = print_gpt_options()

    import testLoans

    task = testLoans.set_task(0)
    context = testLoans.set_context(0)
    instructions = testLoans.set_loan_instructions(prompt_key)
    persona = testLoans.set_persona(chosen_persona)
    
    print(chosen_persona)
    testLoans.specific_decisions(testLoans.transformData, chosen_persona)
    testLoans.create_lime(6)
    response = testLoans.get_response(chosen_persona, instructions, chosen_gpt, context)
    print(response)
    exit()

def context2():
    chosen_persona = print_cold_persona()
    prompt_key = print_cold_prompt_options()
    chosen_gpt = print_gpt_options()

    import testLoans

    task = testLoans.set_task(1)
    context = testLoans.set_context(1)
    instructions = testLoans.set_cold_instructions(prompt_key)
    persona = testLoans.set_persona(chosen_persona)
    testLoans.set_persona(persona)
    response = testLoans.get_cold_response(persona, instructions, chosen_gpt, context)
    print(response)
    exit()

def context3():
    chosen_persona = print_passFail_persona()
    prompt_key = print_passFail_prompt_options()
    chosen_gpt = print_gpt_options()

    import testLoans

    
    context = testLoans.set_context(2)
    instructions = testLoans.set_passFail_instructions(prompt_key)
    persona = testLoans.set_persona(chosen_persona)
    response = testLoans.get_passFail_response(persona, instructions, chosen_gpt, context)
    print(response)
    exit()




def print_gpt_options():
    option_chosen = 0

    print()
    print("Please make a selection:")
    print('\n')
    for key in gpt_options.keys():
        print(str(key) + '.', gpt_options[key], end = "  ")
        print('\n')
        print('―' * 100)
    print()
    print("End of top-level options")
    print()
    option = ' '
    try: 
        option = int(input('Enter your choice: '))
    except KeyboardInterrupt: 
        print('Interrupted')
        exit()
    except: 
        print('Wrong input option. Please enter a number ...') 
    if option == 1:
        option_chosen = 1
    elif option == 2:
        option_chosen = 2 
    else: 
        option_chosen = 1
    return option_chosen


def print_loan_prompt_options():
    option_chosen = 0

    print()
    print("Please make a selection:")
    print('\n')
    for key in loan_prompt_options.keys():
        print(str(key) + '.', loan_prompt_options[key], end = "  ")
        print('\n')
        print('―' * 100)
    print()
    print("End of top-level options")
    print()
    option = ' '
    try: 
        option = int(input('Enter your choice: '))
    except KeyboardInterrupt: 
        print('Interrupted')
        exit()
    except: 
        print('Wrong input option. Please enter a number ...') 
    if option == 1:
        option_chosen = 1
    elif option == 2:
        option_chosen = 2 
    elif option == 3:
        option_chosen = 3
    elif option == 4: 
        option_chosen = 4
    elif option == 5: 
        option_chosen = 5
    elif option == 6:
        option_chosen = 6 
    elif option == 7: 
        option_chosen = 7
    elif option == 8:
        option_chosen = 8 
    else: 
        option_chosen = 0
    return(option_chosen)

def print_cold_prompt_options():
    option_chosen = 0

    print()
    print("Please make a selection:")
    print('\n')
    for key in cold_prompt_options.keys():
        print(str(key) + '.', cold_prompt_options[key], end = "  ")
        print('\n')
        print('―' * 100)
    print()
    print("End of top-level options")
    print()
    option = ' '
    try: 
        option = int(input('Enter your choice: '))
    except KeyboardInterrupt: 
        print('Interrupted')
        exit()
    except: 
        print('Wrong input option. Please enter a number ...') 
    if option == 1:
        option_chosen = 1
    elif option == 2:
        option_chosen = 2 
    elif option == 3:
        option_chosen = 3
    elif option == 4: 
        option_chosen = 4
    elif option == 5: 
        option_chosen = 5
    elif option == 6:
        option_chosen = 6 
    elif option == 7: 
        option_chosen = 7
    elif option == 8:
        option_chosen = 8 
    elif option == 9: 
        option_chosen = 9
    else: 
        option_chosen = 0
    return(option_chosen)

def print_cold_persona():
    option_chosen = 0

    print()
    print("Please make a selection:")
    print('\n')
    for key in cold_personas.keys():
        print(str(key) + '.', cold_personas[key], end = "  ")
        print('\n')
        print('―' * 100)
    print()
    print("End of top-level options")
    print()
    option = ' '
    try: 
        option = int(input('Enter your choice: '))
    except KeyboardInterrupt: 
        print('Interrupted')
        exit()
    except: 
        print('Wrong input option. Please enter a number ...') 
    if option == 1:
        option_chosen = 1
    elif option == 2: 
        option_chosen = 2
    elif option == 3: 
        option_chosen = 3
    elif option == 4: 
        option_chosen = 4
    else: 
        print("ERROR: input an integer number")

    return option_chosen

def print_passFail_persona():
    option_chosen = 0

    print()
    print("Please make a selection:")
    print('\n')
    for key in passFail_personas.keys():
        print(str(key) + '.', passFail_personas[key], end = "  ")
        print('\n')
        print('―' * 100)
    print()
    print("End of top-level options")
    print()
    option = ' '
    try: 
        option = int(input('Enter your choice: '))
    except KeyboardInterrupt: 
        print('Interrupted')
        exit()
    except: 
        print('Wrong input option. Please enter a number ...') 
    if option == 11:
        option_chosen = 11
    elif option == 22: 
        option_chosen = 22
    elif option == 33: 
        option_chosen = 33
    elif option == 44: 
        option_chosen = 44
    else: 
        print("ERROR: input an integer number")

    return option_chosen

def print_passFail_prompt_options():
    option_chosen = 0

    print()
    print("Please make a selection:")
    print('\n')
    for key in passFail_prompt_options.keys():
        print(str(key) + '.', passFail_prompt_options[key], end = "  ")
        print('\n')
        print('―' * 100)
    print()
    print("End of top-level options")
    print()
    option = ' '
    try: 
        option = int(input('Enter your choice: '))
    except KeyboardInterrupt: 
        print('Interrupted')
        exit()
    except: 
        print('Wrong input option. Please enter a number ...') 
    if option == 1:
        option_chosen = 1
    elif option == 2:
        option_chosen = 2 
    elif option == 3:
        option_chosen = 3
    elif option == 4: 
        option_chosen = 4
    elif option == 5: 
        option_chosen = 5
    elif option == 6:
        option_chosen = 6 
    elif option == 7: 
        option_chosen = 7
    elif option == 8:
        option_chosen = 8 
    elif option == 9: 
        option_chosen = 9
    else: 
        option_chosen = 0
    return(option_chosen)


def print_persona_options():
    persona = 0
    print_menu()
    option = ''
    try:
        option = int(input('Enter your choice: '))
    except KeyboardInterrupt:
        print('Interrupted')
    except:
        print('Wrong input. Please enter a number ...')

        # Check what choice was entered and act accordingly
    if option == 1:
        #print("Option 1: Scheana Selected")
        #option1()
        persona = 6
    elif option == 2: 
        #print("Option 2: Ariana Selected")
        #option2()
        persona = 7
    elif option == 3: 
        #print("Option 3: Tom Selected")
        #option3()
        persona = 18
    elif option == 4: 
       # print("Option 4: Jax Selected")
        #option4()
        persona = 42
    elif option == 5: 
        #print("Option 5: James Selected")
        #option5()
        persona = 38
    elif option == 6: 
        #print("Option 6: Rob Selected")
        #option6()
        persona = 70
    else:
        print('Invalid option. Please enter a number between 1 and 6.')  
    return persona 
      


if __name__=='__main__':
    while(True):
        print()
        print("Please make a selection:")
        print('\n')
        for key in context_options.keys():
            print(str(key) + '.', context_options[key], end = "  ")
            print('\n')
            print('―' * 100)
        print()
        print("End of top-level options")
        print()
        option = ' '
        try: 
            option = int(input('Enter your choice: '))
        except KeyboardInterrupt: 
            print('Interrupted')
            exit()
        except: 
            print('Wrong input option. Please enter a number ...') 
        if option == 1:
            context1()
        elif option == 2:
            context2()
        elif option == 3: 
            context3()
        else: 
            exit()



### MAKE THE GENERIC PROMPT WITH ZERO SHOT PROMPTING --> NO INSTRUCTIONS OR PERSONA 