# Helper classes that can be used across all scripts or strategies

import time


class Timer:
    '''The application flow logic should be dependent on 'self.timer_is_active' flag
    The timer duration may pass but some functions may need to start 
    at some given point after that moment
    Timer life cycle:
    1. .start_and_enable_timer_is_active_flag(duration)
    2. any logic dependent on timer is activated if '.timer_is_active' is True
    3. .deactivate_if_time_passed() MUST be executed to disable '.timer_is_active' flag

    Another approach is just to add one or many timers to a list. 
    Create a new timer instance, start it with start_and_enable_timer_is_active_flag 
    and append to the timers list.
    Later each timer from the list is checked with '.has_time_passed' and
    when the timer is over it's just deleted from the list''' 
       
    def __init__(self, name="Anonymous", duration=0, activate=True):
        self._name = name
        if activate:
            self._start_time = int(time.time() * 1000)
            self._duration = duration  # Set the duration for the timer
            self._timer_is_active = True  # Timer becomes active when started            
        else:
            self._start_time = None # time in milliseconds
            self._duration = 0  # duration in milliseconds
            self._timer_is_active = False  # Timer is inactive by default
    
    @property
    def name(self):
        return self._name    
    
    @property
    def timer_is_active(self):
        return self._timer_is_active

    def start_and_enable_timer_is_active_flag(self, duration: int):
        '''Start the timer and enable the active flag.
        
        Args:
            duration (int): Duration of the timer in milliseconds.
        '''
        # Set the current time in milliseconds and activate the timer
        self._start_time = int(time.time() * 1000)
        self._duration = duration  # Set the duration for the timer
        self._timer_is_active = True  # Timer becomes active when started
    
    def has_time_passed(self):
        if self._start_time is not None and self._duration is not None:
            # Get the current time in milliseconds
            current_time = int(time.time() * 1000)
            # Check if the specified duration has passed        
            if current_time - self._start_time >= self._duration:
                return True
        return False

    def deactivate_if_time_passed(self):
        if self.has_time_passed():
            self._timer_is_active = False  # Deactivate the timer if time has passed
            self._start_time = None
            self._duration = None  # Reset the duration and start time
        return self._timer_is_active               
