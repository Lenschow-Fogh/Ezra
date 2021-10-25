import spacy

text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."
mortText = "Morten LenschowHansen SOFTWARE TECHNOLOGY STUDENT Aarhus University, Finlandsgade 22, 8000 Aarhus, Denmark  (+45) 27 79 96 69 |  hansenmann2497@hotmail.com Unity Technologies May 7, 2020 NIELS HEMMINGSENS GADE 24 1153 COPENHAGEN Application for the internship (Fall 2020) Dear Sir/Madam About Me I am currently on my fourth semester of my Bachelor in Software Engineering at Aarhus University, Denmark. My current aim is achieving a Masters degree in computer science. In my free time I like to play games or soccer with friends, or follow sports such as F1 and eSports (CS:GO & RLCS). I am a very social guy, I love to be around people and I tend to be funny and outgoing. Why Unity Technologies? Unity Technologies provides a innovative platform for creating games to businesses or consumers to reach targeted audiences. I am very keen on becoming part of this process. On my current semester I have developed a mobile application (reference to Fruit Fly in attached CV). Also I have worked on several other projects like writing code for the logic, data binding and flow of programs. This has given me experience on non-game applications. I thoroughly enjoyed working with this, and it ensured me that it is something I want to continue working with, therefore Unity Technologies. Why me? Professionally I have a lot to offer, I have great experience with C# and other OOP languages, design principles (such as SOLID), software test and GUI-applications (full stack). Process-wise I have worked in project teams and gained experience with SCRUM. I can easily communicate with everyone on/or off the team and I like to work in teams. I always strive to be the best I can be, and that goes for my work/study as well, where I always try to deliver the best results. I am looking forward to your reply. Sincerely, Morten Lenschow Hansen MORTEN LENSCHOW HANSEN"

nlp = spacy.load("en_core_web_trf")
doc = nlp("Morten LenschowHansen SOFTWARE TECHNOLOGY STUDENT Aarhus University, Finlandsgade 22, 8000 Aarhus, Denmark  (+45) 27 79 96 69 |  hansenmann2497@hotmail.com Unity Technologies May 7, 2020 NIELS HEMMINGSENS GADE 24 1153 COPENHAGEN Application for the internship (Fall 2020) Dear Sir/Madam About Me I am currently on my fourth semester of my Bachelor in Software Engineering at Aarhus University, Denmark. My current aim is achieving a Masters degree in computer science. In my free time I like to play games or soccer with friends, or follow sports such as F1 and eSports (CS:GO & RLCS). I am a very social guy, I love to be around people and I tend to be funny and outgoing. Why Unity Technologies? Unity Technologies provides a innovative platform for creating games to businesses or consumers to reach targeted audiences. I am very keen on becoming part of this process. On my current semester I have developed a mobile application (reference to Fruit Fly in attached CV). Also I have worked on several other projects like writing code for the logic, data binding and flow of programs. This has given me experience on non-game applications. I thoroughly enjoyed working with this, and it ensured me that it is something I want to continue working with, therefore Unity Technologies. Why me? Professionally I have a lot to offer, I have great experience with C# and other OOP languages, design principles (such as SOLID), software test and GUI-applications (full stack). Process-wise I have worked in project teams and gained experience with SCRUM. I can easily communicate with everyone on/or off the team and I like to work in teams. I always strive to be the best I can be, and that goes for my work/study as well, where I always try to deliver the best results. I am looking forward to your reply. Sincerely, Morten Lenschow Hansen MORTEN LENSCHOW HANSEN")


print("------------- PRINTING ALL ENTITIES -------------")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
print("---------- FINISHED PRINTING ENTITIES ------------")



def censor_word(text, word):
    print("censor_word:", word)
    print("censor_word text:", text)
    censoredWord = "*" * len(word)
    return text.replace(word, censoredWord)

def censor_ents(doc, originalText):
    if doc.ents:
        censoredText = originalText
        for ent in doc.ents:
            label = ent.label_
            if label == "PERSON" or label == "GPE":
                censoredText = censor_word(originalText, ent.text)
    return censoredText