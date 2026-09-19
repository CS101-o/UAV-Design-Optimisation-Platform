class Node: 
    def __init__(self, value):
        self.value = value 
        self.children = []

def maxNode(node,depth, alpha,  beta):   
    if depth == 0 and node.children is not None:
        return node.value 
    else:
        Best = -float('inf') # setting best to - inifinity 
    for i in node.children:
        val = minNode(i, depth -1, alpha, beta )
        Best = max(val, Best)
        alpha = max(alpha, Best)
        if alpha >= beta:
            break 

        return Best
        
def minNode(node, depth, alpha, beta):
    if depth == 0 and node.children is not None:
        return node.value 
    else:
        Best = float("inf") # setting best to +inifinity 
        for i in node.children:
            val = maxNode(i, depth -1, alpha, beta)
            Best = min(val, Best)
            beta = min(alpha, Best)
            if alpha >= beta:
                break 

        return Best

def main(tree, depth, player): 
    beta = float('inf')
    alpha = -float('inf')
    if player == "white":
        return maxNode(tree, depth , alpha, beta)
    else:
        return minNode(tree, depth , alpha, beta)
        
node = Node(10)
children = node.children.append(Node(2))
children = node.children.append(Node(4))
children = node.children.append(Node(20))
print(node.children)
result = main(node, 1, "white" )

    



