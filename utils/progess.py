class Progress:
    def __init__(self, total_rounds):
        self.total_rounds = total_rounds
        self.total_train_duration = 0

    def step(self, step_duration):
        self.total_train_duration += step_duration

    def eta(self, epoch_num):
        remaining_train_duration = self.total_train_duration / (epoch_num + 1) * (self.total_rounds - epoch_num - 1)

        hours = int(remaining_train_duration // 3600)
        minutes = int(remaining_train_duration % 3600 // 60)

        return f'{hours} hrs {minutes} mins remaining'
