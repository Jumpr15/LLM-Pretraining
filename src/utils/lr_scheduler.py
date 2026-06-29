from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR

class WSD_Scheduler():
     def __init__(self, warmup_steps, iterations, optimizer, decay_ratio):
          
          self.warmup_steps = warmup_steps
          self.iterations = iterations
          self.decay_ratio = decay_ratio
          
          warmup_scheduler = LinearLR(
               optimizer,
               start_factor=0.1,
               end_factor=1.0,
               total_iters=self.warmup_steps
          )

          stable_scheduler = ConstantLR(
               optimizer,
               factor=1.0
          )

          cosine_decay_scheduler = CosineAnnealingLR(
               optimizer, 
               T_max=self.iterations*self.decay_ratio
          )

          self.wsd_scheduler = SequentialLR(
               optimizer,
               schedulers=[warmup_scheduler, stable_scheduler, cosine_decay_scheduler],
               milestones=[self.warmup_steps, self.iterations * (1 - self.decay_ratio)]
          )
          
     def get_scheduler(self):
          return self.wsd_scheduler