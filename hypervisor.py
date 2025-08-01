import os
import ppo
import environment

modelDir = "./models/"
states = []

def main():

    print("-"*20)
    print("1: Create")
    print("2: Resume")
    print("3: Preview")
    print("4: List")
    choice = input("choice: ")

    match int(choice):
        case 1:
            create()
        case 2:
            resume()
        case 3:
            preview()
        case 4:
            listModels()
    
def listModels(verbose=True):

    items = os.listdir(modelDir)
    index = 0
    if (verbose):
        print("Available models")
        for item in items:
            print(str(index) + ": " + str(item))
            index+=1
    return(items)

def create():
    
    models = listModels(False)
    index = 0
    for item in models:
        models[index] = item[6:-4]
        # print(models[index])
        index+=1
    
    name = ""
    while (name == ""):
        name = input("Model name: ")

        if (name in models):
            name = ""
            print("Model already exists")
        if (" " in name):
            name = ""
            print("Model name must not have a space in it")

    seed = ""
    while (seed == ""):
        seed = input("Simulation seed(default 11111111): ")
        
        if (seed == ""):
            seed = str(11111111)
        
        if (len(seed) != 8):
            seed = ""
            print("Seed must be 8 numbers")
        
        try:
            int(seed)
        except:
            seed = ""
            print("seed must be only numbers")

    env = ppo.ppo()
    states.append([name,env])
    print("Starting training...")
    manage(env, name, seed, True)
    

def resume():
    model = None
    models = listModels()

    modelIndex = None
    while (modelIndex == None):
        modelIndex = int(input("Select a model index: "))

        try:
            model = models[modelIndex]
        except IndexError:
            modelIndex = None
            print("Model does not exist...")
        
    seed = ""
    while (seed == ""):
        seed = input("Simulation seed(default 11111111): ")
        
        if (seed == ""):
            seed = str(11111111)
        
        if (len(seed) != 8):
            seed = ""
            print("Seed must be 8 numbers")
        
        try:
            int(seed)
        except:
            seed = ""
            print("seed must be only numbers")

    name = model[6:-4]
    env = ppo.ppo()
    states.append([name,env])
    print("Starting training...")
    manage(env, name, seed, False)
    

def manage(env, name, seed, newState):
    
    env.__init__(seed=seed)
    
    new = newState
    cnt = 0
    while (not env.ppo_done):

        if (cnt >= 30):
            # prevent loop run off
            break
        # update episode count
        episode = env.currentEpisode

        if (new):
            new = False
            try:
                # try running the simulation
                env.run_ppo(name=name)
            except:
                cnt+=1
                continue
        else:
            # if it has crashed, handle restarting it
            try:
                env.__init__(seed=seed, currentEpisode=episode)
                env.run_ppo(name=name, load=True)
            except:
                cnt+=1
                continue

def preview():
    model = None
    models = listModels()

    modelIndex = None
    while (modelIndex == None):
        modelIndex = int(input("Select a model index: "))

        try:
            model = models[modelIndex]
        except IndexError:
            modelIndex = None
            print("Model does not exist...")

    name = model[6:-4]

    seed = ""
    while (seed == ""):
        seed = input("Simulation seed(default 11111111): ")
        
        if (seed == ""):
            seed = str(11111111)
        
        if (len(seed) != 8):
            seed = ""
            print("Seed must be 8 numbers")
        
        try:
            int(seed)
        except:
            seed = ""
            print("seed must be only numbers")
    
    env = ppo.ppo()
    states.append([name,env])

    env.__init__(seed=seed)
    env.env_train = environment.IoaiNavEnv(headless=False, seed=seed)

    env.run_ppo(name, load=True)
    


#preview("test", 11111111)
#manage("test",11111111)
main()