import clips

clp_env = ''

def load_clp():
    print("clp_env loaded for clips")
    global clp_env
    clp_env = clips.Environment()
    clp_env.load('D:\pythonProject\constructs_car.clp')


def get_decision(sign, distance, speed):
    global clp_env
    template = clp_env.find_template('state')
    fact = template.assert_fact(sign=sign,
                                 distance=distance,
                                 speed=speed)
    result = 1
    clp_env.run()
    for fact in clp_env.facts():
        fact = str(fact)
        if 'result_' in str(fact):
            fact = fact.replace("(result_speed:", "")
            result = fact[:-1].replace(" ","")
    clp_env.reset()
    return result
