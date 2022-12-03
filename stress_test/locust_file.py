from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    wait_time = between(0.3, 1)
    
    def on_start(self):
        '''
        self.client.post("/login", {
            "username": "test_user",
            "password": ""
        })
        '''
    
    @task
    def status(self):
        self.client.get("/status")
        
    @task
    def threshold_on(self):
        self.client.get("/threshold_on")

    @task
    def threshold_off(self):
        self.client.get("/threshold_off")
