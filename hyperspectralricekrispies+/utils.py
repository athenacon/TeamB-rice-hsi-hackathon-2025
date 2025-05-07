import datetime

def raise_(e: Exception) -> None:
    """
    Raise an error e. Made for lambdas
    """
    raise e

def print_duck(finished_training):
    if finished_training:
        print(" ____________________ ") # Made using cowsay
        print("< Finished training! >") # echo "Finished training!" | cowsay -f duck
        print(" -------------------- ")
        print(" \\                   ")
        print("  \\                  ")
        print("   \\ >()_            ")
        print("      (__)__ _        ")
    else:
        print(" ____         ")
        print("< ?! >        ")
        print(" ----         ")
        print(" \\           ")
        print("  \\          ")
        print("   \\ >()_    ")
        print("      (__)__ _")

def get_current_iso_datetime() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()