import matplotlib.pyplot as plt
from IPython import display

#plt.ion()

fig, ax = plt.subplots()
def plot_old(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def plot(scores, mean_scores):
    plt.figure(figsize=(12, 6))

    # Clear the current figure's content
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')

    # Plot both scores and mean_scores
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')

    plt.legend()

    # Choosing whether you want to display the plot or
    # save it by uncommenting one of the following commands:

    # Saves the figure
    # plt.savefig('plot.png')

    # Display
    plt.draw()
    plt.pause(1)  # pause for a short period

    plt.show(block=False)
