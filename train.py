"""Train the final velocity/RGB-MSE model on the fixed KAUSTics split."""
from kausticsnet.engine import main

if __name__ == '__main__':
    main(finetune=False)
