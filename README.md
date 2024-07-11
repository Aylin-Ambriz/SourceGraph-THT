# Take Home Task - SourceGraph

This project takes an input of natural language and outputs python code to achieve this task. It was a daunting task when presented, but in hindsight it was an excellent journey of learning. Starting from zero, I pondered about the architecture and dataset required to accomplish this. It was not a simple decoder-decoder, there was more layers that needed to be implemented. And given my at home compute restrictions, I needed to work within those limitations. I decided to start with data, my first idea being to create synthetic data, but that approach came with many problems - a lack of high-quality synthetic data as well as not being able to produce enough within the time frame. So I researched datasets that could work and came across the Google Research LLM data which contained a json of natural language to python code. With a very high quality dataset in hand, I could now tackle the architecture. I had a decoder-decoder base, but the data had to do was understand natural language, so I decided to use Google T5 model to train it - its use of vectors as opposed to tensors would ensure it understood English semantics and was within my compute power. The next step was for it to understand code logic, so I decided to use Salesforce CodeT5 - it could train the decoder to understand coding logic. I then went on to process the data, implement these models along with decoder decoder architecture which resulted in our final output. 


Proudly created by Atharva Patel

