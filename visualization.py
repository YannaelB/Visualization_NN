import csv
import tensorflow as tf
from tensorflow import keras
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import render



    # This function is very slow and will make your code run very slowly. If you want to make it faster, 
    # you can modify a bit the node_states.function in order to save each graph state in an array. Only at the end, you will call draw_ann.function to 
    # create images and video for all graph states, but not while your code is running, otherwise it's slowing down the code.
def create_png_from_model(model_file,input,filenames,n,pict_file):

    # load the model you want to visualise
    model = keras.models.load_model(model_file)
    #model.summary()

        # Create a vefctor with the structure of the model
    nb_neurones = np.zeros((len(model.layers)+1)) 
    nb_neurones[0] = len(model.layers[0].get_weights()[0]) #equivalent to input_shape = model.layers[0].input_shape[1:]

    for i in range(1,len(model.layers)+1):
        nb_neurones[i] = model.layers[i-1].output_shape[-1]


        # Calculate the value of each neurons also called nodes here (cause we use graph-package)
    def node_states(input):
        def sigm(x):
            return 1/(1 + np.exp(-x))
        def Relu(x):
            return max(0,x)
        node_states_tab = [input]
        node_states_tab_normalized = input
        for i in range(1,len(model.layers)+1):
            layer_node_values = []
            for j in range(int(nb_neurones[i])):
                node_value_j = 0
                for k in range(int(nb_neurones[i-1])):
                    # Calculate the value of each neurone
                    node_value_j += node_states_tab[i-1][k]*model.layers[i-1].get_weights()[0][k][j]      
                # Add the neurone bias's
                node_value_j += model.layers[i-1].get_weights()[1][j] 
                layer_node_values.append(Relu(node_value_j)) # Applicate the activation function to the neurone (here i use ReLu function)

            node_states_tab += [layer_node_values]
            #normalize each neurone layer
            if max(layer_node_values) != 0:
                layer_node_values = [i/max(layer_node_values) for i in layer_node_values]
            node_states_tab_normalized += layer_node_values

        return node_states_tab, node_states_tab_normalized


        # Create png of the state of the model for one input state
    def draw_ann_from_csv():
        
        def sigm(x):
            return 1/(1 + np.exp(-x))
            
        # Load NN_structure of the model from the .csv file
        df = pd.read_csv(f'{model_file}.csv')

        # Create a graph which will represent the NN
        G = nx.DiGraph()
        G.add_nodes_from(df['source'].unique().tolist() + df['target'].unique().tolist()) # Create nodes
        G.add_edges_from(df[['source', 'target']].values.tolist()) # Create edges

        # Color each neurone according to their value
        node_colors = {}
        i = 0
        node_states_tab, node_states_tab_normalized = node_states(input)
        for node in G.nodes():

                # Color each node according to their rate (between 0 and 1)
            node_colors[node] = (0, node_states_tab_normalized[i], 0)

                # Use a different color for the neurone which predict the output (max neurone of the output_layer)
            length_output = len(node_states_tab[-1])
            prediction_output = len(node_states_tab_normalized) - (length_output - node_states_tab[-1].index(max(node_states_tab[-1]))-1 )
            if i == prediction_output-1:
                node_colors[node] = (node_states_tab_normalized[i], 0, 0)
            i +=1
            

           # Color edges. Here i use a static color cause the agent is already trained so the weigth are fixe 
        edge_colors = {}
        j = 0
        for (u, v, d) in G.edges(data=True):
                # Collect weigth value's for each edges/connection
            weight = df['weight'].unique().tolist()[j]
                # Create a list of colors
            edge_colors[(u, v)] = (0.95*sigm(weight), 0.95*sigm(weight), 0.95*sigm(weight)) #Here I used sigmoid to represent weight cause i needed a function from R to [0,1] for colouring the weight. 
            # I multiply per 0.95 to avoid pure white color
            j +=1

        
            # Applied a fixe and layered position to the graph
        pos = nx.nx_pydot.pydot_layout(G)
        pos = nx.nx_pydot.pydot_layout(G, prog="dot")
        nx.draw_networkx_nodes(G, pos,node_color=list(node_colors.values()))
        nx.draw_networkx_edges(G, pos,edge_color=list(edge_colors.values()), width=0.5)

        
        # Visualize the graph NN
        plt.axis('off')
        #plt.show()

            # save frame
        filename = f"{pict_file}/{model_file}{n}.png"
        filenames.append(filename)
        plt.savefig(filename,dpi = 100)
        plt.close()
        print("image's done")
    draw_ann_from_csv()
    return None



def create_csv_from_model(model_file):

    # load the model you want to visualise
    model = keras.models.load_model(model_file)
    #model.summary()

        # Create a vector with the structure of the model
    nb_neurones = np.zeros((len(model.layers)+1)) 
    nb_neurones[0] = len(model.layers[0].get_weights()[0]) #equivalent to input_shape = model.layers[0].input_shape[1:]

    for i in range(1,len(model.layers)+1):
        nb_neurones[i] = model.layers[i-1].output_shape[-1]
    print("Structure of your model :",nb_neurones)


        # Create a file.CSV of the model for visualizing the model with graph module (NetworkX)
    def create_ann_csv():
        # Define the structure of the NN
        layers = ['input']
        for i in range(1,len(model.layers)):
            layers.append(f'hidden{i}')
        layers.append('output')
        
        # Use a vector which reflect the structure of the model
        nodes_per_layer = nb_neurones
        
        # Création du fichier CSV
        with open(f'{model_file}.csv', mode='w', newline='') as file:
            fieldnames = ['source', 'target','length','weight']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({'source': "source", 'target': "target",'length':"length",'weight':"weight"})

            # Add edges/connections between nodes
            for i in range(len(layers)-1):
                for j in range(int(nodes_per_layer[i])):
                    for k in range(int(nodes_per_layer[i+1])):
                        source = f'{layers[i]}{j+1}'
                        target = f'{layers[i+1]}{k+1}'
                        length = f'1'
                        weight = model.layers[i].get_weights()[0][j][k]
                        writer.writerow({'source': source, 'target': target,'length':length,'weight':weight})
            print("csv done")

    create_ann_csv()