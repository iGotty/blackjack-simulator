
from blackjack import BlackJackGame
from dealer import HouseRules



from blackjack import BlackJackGame
from dqn_agent import DQNAgent
import numpy as np
from collections import defaultdict

# Definir los parámetros del entorno y del agente
state_size = 4  # Ejemplo: Valor de la mano del jugador, carta del dealer, true count, bankroll
action_size = 3  # HIT, STAND, DOUBLE (simplificando las acciones iniciales)
batch_size = 32
n_episodes = 10

# Diccionario para registrar decisiones por true count (-4 a 4)
decision_table = {
    tc: defaultdict(lambda: {"HIT": 0, "STAND": 0, "DOUBLE": 0, "SPLIT": 0}) for tc in range(-4, 5)
}
# Para valores fuera de rango -4 y 4
decision_table['4+'] = defaultdict(lambda: {"HIT": 0, "STAND": 0, "DOUBLE": 0, "SPLIT": 0})
decision_table['4-'] = defaultdict(lambda: {"HIT": 0, "STAND": 0, "DOUBLE": 0, "SPLIT": 0})

# Inicializar el entorno y el agente DQN
agent = DQNAgent(state_size, action_size)
# Definir las reglas de la casa como un objeto HouseRules
house_rules = HouseRules(standValue=17, DASoffered=True, RSAoffered=False, LSoffered=False, doubleOnSoftTotal=True)

# Inicializar el entorno del juego BlackJackGame correctamente
game = BlackJackGame(shoeSize=6, bankroll=1000, hands=1000, tableMin=10, penetration=0.5, houseRules=house_rules)

# Función para clasificar true count en el rango adecuado
def classify_true_count(tc):
    if tc < -4:
        return '4-'
    elif tc > 4:
        return '4+'
    return tc

for e in range(n_episodes):
    # Reiniciar el juego
    print(f"Starting episode {e + 1}/{n_episodes}")  # Agrega este print para verificar que comienza cada episodio
    game.startGame()
    done = False
    while not done:
        for player in game.players:
            print(f"Processing player {player.name} in episode {e + 1}")  # Este print te mostrará el progreso del jugador
            if player.name != "DQN Agent":
                continue #Salta los jugadores que no son el agente DQN
            state = game.get_state(player)
            state_arr = [state["player_hand_value"], state["dealer_upcard_value"], state["true_count"], state["player_bankroll"]]

            # El agente toma una acción
            action_index = agent.act(state_arr)
            action_str = ["HIT", "STAND", "DOUBLE"][action_index]  # Convertimos el índice en la acción

            # Ejecutar la acción en el simulador y obtener el nuevo estado y recompensa
            next_state = game.get_state(player)
            reward = player.reward  # Recompensa de ganancias/pérdidas
            next_state_arr = [next_state["player_hand_value"], next_state["dealer_upcard_value"], next_state["true_count"], next_state["player_bankroll"]]

            # Clasificar true count
            tc_class = classify_true_count(next_state["true_count"])
            
            # Determinar el tipo de mano
            if player.getStartingHand().isPair():
                decision_table[tc_class]["SPLIT"][action_str] += 1
            elif player.getStartingHand().isSoftTotal(0):
                decision_table[tc_class]["SOFT"][action_str] += 1
            else:
                decision_table[tc_class]["HARD"][action_str] += 1

            # Guardar la experiencia y entrenar el agente
            done = True if player.bankroll <= 0 else False
            agent.remember(state_arr, action_index, reward, next_state_arr, done)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    if (e + 1) % 100 == 0:
        print(f"Episode {e + 1}/{n_episodes} - Epsilon: {agent.epsilon}")

# Mostrar las tablas de decisiones
print("Tabla de decisiones por true count (-4 a +4):")
for tc, actions in decision_table.items():
    print(f"True Count {tc}:")
    for hand_type, hand_actions in actions.items():
        print(f"  {hand_type} hands:")
        total = sum(hand_actions.values())
        for action, count in hand_actions.items():
            prob = count / total if total > 0 else 0
            print(f"    {action}: {prob:.2f}")
