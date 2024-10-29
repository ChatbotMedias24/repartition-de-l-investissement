import streamlit as st
import openai
import streamlit as st
from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message  # Importez la fonction message
import toml
import docx2txt
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import docx2txt
from dotenv import load_dotenv
if 'previous_question' not in st.session_state:
    st.session_state.previous_question = []

# Chargement de l'API Key depuis les variables d'environnement
load_dotenv(st.secrets["OPENAI_API_KEY"])

# Configuration de l'historique de la conversation
if 'previous_questions' not in st.session_state:
    st.session_state.previous_questions = []

st.markdown(
    """
    <style>

        .user-message {
            text-align: left;
            background-color: #E8F0FF;
            padding: 8px;
            border-radius: 15px 15px 15px 0;
            margin: 4px 0;
            margin-left: 10px;
            margin-right: -40px;
            color:black;
        }

        .assistant-message {
            text-align: left;
            background-color: #F0F0F0;
            padding: 8px;
            border-radius: 15px 15px 15px 0;
            margin: 4px 0;
            margin-left: -10px;
            margin-right: 10px;
            color:black;
        }

        .message-container {
            display: flex;
            align-items: center;
        }

        .message-avatar {
            font-size: 25px;
            margin-right: 20px;
            flex-shrink: 0; /* Empêcher l'avatar de rétrécir */
            display: inline-block;
            vertical-align: middle;
        }

        .message-content {
            flex-grow: 1; /* Permettre au message de prendre tout l'espace disponible */
            display: inline-block; /* Ajout de cette propriété */
}
        .message-container.user {
            justify-content: flex-end; /* Aligner à gauche pour l'utilisateur */
        }

        .message-container.assistant {
            justify-content: flex-start; /* Aligner à droite pour l'assistant */
        }
        input[type="text"] {
            background-color: #E0E0E0;
        }

        /* Style for placeholder text with bold font */
        input::placeholder {
            color: #555555; /* Gris foncé */
            font-weight: bold; /* Mettre en gras */
        }

        /* Ajouter de l'espace en blanc sous le champ de saisie */
        .input-space {
            height: 20px;
            background-color: white;
        }
    
    </style>
    """,
    unsafe_allow_html=True
)
# Sidebar contents
textcontainer = st.container()
with textcontainer:
    logo_path = "medi.png"
    logoo_path = "NOTEPRESENTATION.png"
    st.sidebar.image(logo_path,width=150)
   
    
st.sidebar.subheader("Suggestions:")
questions = [
    "Donnez-moi un résumé du rapport ",
        "Quels types d'infrastructures sont ciblés par le projet de loi de finances de 2025 ?",
        "Quelles sont les principales réalisations des programmes d'investissement entre 2021 et 2024 ?",      
        "Comment le projet de loi de finances 2025 prévoit-il de réduire les disparités régionales en termes d'investissement ?",
        """Quels sont les objectifs du programme "Ville sans Bidonvilles" (VSB) pour 2025 ?"""

# Initialisation de l'historique de la conversation dans `st.session_state`
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = StreamlitChatMessageHistory()
def main():
    text=r"""
    
INTRODUCTION

Dans un contexte géopolitique bouleversé, l'économie mondiale fait face à des défis
d'une grande ampleur. Les Gouvernements ont été contraints de resserrer leurs
politiques monétaires, ce qui a entraîné un ralentissement significatif de la
consommation et de l'investissement et a accentué la stagnation des activités
économiques.

En effet, les tensions géopolitiques, notamment les guerres en Ukraine et au
Moyen-Orient, adossées aux sanctions économiques et aux restrictions commerciales
qui en ont découlé, ont perturbé davantage les chaînes d'approvisionnement
mondiales et provoqué une hausse des coûts des matières premières. Cette situation a
entraîné des pressions inflationnistes et a réduit le pouvoir d'achat des
consommateurs, principalement dans les pays en voie de développement, notamment
en Afrique.

Dans ce contexte défavorable, l’économie marocaine a subi les effets des
bouleversements économiques au niveau international, bien qu'affichant une résilience
notable au premier semestre de l’année 2024. Cette résilience se manifeste
particulièrement à travers la redynamisation du secteur touristique, qui s’impose
comme un moteur clé de l’économie nationale. En effet, le Royaume a accueilli un
nombre record de touristes durant les huit premiers mois de l’année 2024, estimé à
11,8 millions de touristes, confirmant ainsi la compétitivité et l’attractivité renouvelées
de la destination Maroc. Ce dynamisme contribue non seulement à la création
d’emplois, mais aussi à l’accroissement des recettes en devises, tout en stimulant des
secteurs connexes tels que l’hôtellerie, la restauration et l’artisanat, ce qui consolide
l’impact socio-économique du tourisme sur la croissance économique nationale.

Parallèlement, l’investissement dans le secteur énergétique témoigne d'une évolution
significative, à travers le renforcement de la production électrique issue des énergies
renouvelables. Ce développement stratégique permettra de réduire la dépendance
énergétique du Royaume vis-à-vis des importations, tout en promouvant une
croissance durable, fondée sur une transition énergétique résolument tournée vers
l’avenir.

En outre, l'augmentation de 11,9 % des recettes fiscales enregistrée pour les neuf
premiers mois de l’année 2024 par rapport à la même période en 2023, à l’instar de ce
qui a été observé pendant les trois dernières années, témoigne de la dynamique
économique favorable qui a permis de générer des ressources additionnelles pour
l'État, présentant un contexte financier propice, et offrant des marges budgétaires
élargies pour envisager des orientations stratégiques en matière d’investissement,
créatrices d’emplois et de richesse. En effet, cette progression confère à l'État la
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


capacité de déployer des initiatives visant à consolider la croissance économique, tout
en répondant aux exigences de développement durable du pays.

Toutefois, le secteur agricole, pilier essentiel de l’économie nationale, demeure
vulnérable face aux aléas climatiques. Les conditions pluviométriques défavorables et
persistantes, exacerbées par des épisodes de sécheresse récurrentes, ont perturbé les
cycles des cultures, et ont entraîné une baisse notable des rendements. Cette situation
fragilise les revenus des exploitants agricoles et, par extension, déstabilise l’ensemble
de la chaîne de valeur agricole. Les conséquences se manifestent non seulement par
une diminution des ressources disponibles pour les producteurs, mais aussi par une
instabilité accrue des produits alimentaires, impactant par conséquent le tissu
économique rural et augmentant la vulnérabilité de la communauté dépendante de
l'agriculture.

Pour l'année 2025, le Maroc anticipe un raffermissement de l'activité économique,
porté notamment par l’investissement public, avec un renforcement de la dynamique
d’investissement et la création d’opportunités d’emploi. Cela en poursuivant la
construction d’un État social fondé sur une vision inclusive et cohérente.
Parallèlement, le Gouvernement continue de soutenir les investissements privés, dans
le but d'augmenter la part des investissements privés pour qu'ils représentent les deux
tiers de l'investissement total à l’horizon 2035, tel qu’il a été porté par la nouvelle
charte d’investissement récemment adoptée.

Les sections de la présente Note qui accompagne le Projet de Loi de Finances 2025,
proposent de fournir une vue d’ensemble sur la répartition régionale de
l’investissement public dans ce contexte de redressement économique, selon les
principaux secteurs d’intervention des politiques publiques.

La première section de la Note met en relief deux parties dédiées respectivement au
redressement notable de l’économie nationale malgré l’instabilité de l’activité
mondiale, et au renforcement de l'investissement public en tant que catalyseur
stratégique du développement inclusif et durable. La seconde section, quant à elle,
donne un aperçu sur la répartition régionale de l’investissement public, à travers la
présentation des principaux projets structurants relevant des grands secteurs
d’intervention de l’Etat ; à savoir, les secteurs sociaux, les secteurs d’infrastructure et
les secteurs productifs.
                                  NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



PARTIE I : L'INVESTISSEMENT PUBLIC AU SERVICE D’UN
DEVELOPPEMENT INCLUSIF ET DURABLE

En célébrant 25 années de règne de S.M. le Roi, que Dieu l’assiste, le Maroc a relevé le
défi de la modernité et du développement économique durable. Sur ce plan, le
Royaume s’est lancé dans de grands chantiers d’infrastructures et de réformes
institutionnelles structurantes, constituant la clé du succès de sa nouvelle approche de
développement qui repose sur des stratégies sectorielles ambitieuses et diversifiées.
Cette situation a permis à l’économie nationale de faire face aux instabilités
géopolitiques mondiales, et de créer un climat favorable à l’investissement.
1.1 Etats des lieux : Redressement notable de Péconomie nationale malgré
   Pinstabilité de Pactivité mondiale
À l’instar de l'année 2023, durant laquelle l'économie mondiale a enregistré une
croissance modeste de 3,3 %, le premier semestre de l’année 2024 se distingue par
une conjoncture mondiale particulièrement instable, marquée par une prolongation
des tensions géopolitiques ayant engendré des incertitudes économiques prononcées.

Malgré cet environnement international complexe, l’économie marocaine s’affirme en
2024 par une performance notable sur la scène internationale, renforçant son
attractivité auprès des investisseurs étrangers. La stabilité économique du pays,
marquée par une monnaie stable et des réserves de change importantes, inspire une
confiance croissante à l’international.

Ainsi, le Maroc prévoit une croissance économique de 3,3 % en 2024 qui prend en
compte divers facteurs déterminants. L'inflation, malgré les efforts soutenus du
Gouvernement pour la contenir, demeure relativement élevée, et la stagnation de la
demande extérieure, moteur essentiel de la croissance, limite les exportations et les
recettes issues des échanges internationaux. À cet effet, le Gouvernement a envisagé
des mesures louables pour soutenir le pouvoir d'achat des ménages marocains et
relancer la demande intérieure ; en témoignent les résultats du dialogue social qui a
permis à plus de 1,128 million de fonctionnaires du secteur public de bénéficier d’une
augmentation générale de leurs salaires.

Dans ce contexte, il est important de souligner que la Banque Mondiale a affirmé, dans
son dernier rapport de suivi de la situation économique du Maroc, que le dynamisme
des exportations manufacturières et de services, ainsi que les transferts des marocains
résidant à l’étranger (MRE), ont significativement amélioré le solde courant du pays.

Néanmoins, les défis climatiques demeurent persistants et la sécheresse continue de
sévir, affectant gravement le secteur agricole. En effet, la réduction des rendements
agricoles et la crise hydrique aggravent considérablement la vulnérabilité économique
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


et environnementale du pays. Face à ces défis, le Gouvernement est déterminé à
redoubler les efforts et la vigilance, notamment en ce qui concerne l’impératif de faire
face à la problématique de l'eau, qui s’est accentuée dernièrement par les effets
néfastes des changements climatiques et l'accroissement continu de la demande en
ressources hydriques.

Par ailleurs, et en dépit de ces adversités, le Maroc demeure résolument engagé dans
la mise en oeuvre d'une politique d'investissement public ambitieuse, visant à stimuler
la croissance économique et à affermir la résilience du pays face aux défis actuels.
Cette démarche stratégique, portée par une vision claire de développement, reflète la
volonté du Gouvernement de transformer durablement les fondements socio­
économiques du pays, en s'appuyant sur des investissements d'envergure dans des
secteurs névralgiques.

C’est dans ce cadre que le développement des infrastructures, pierre angulaire de
cette politique, vise à moderniser les réseaux de transport, à diversifier les sources
d'énergie et à optimiser la gestion des ressources hydriques. Les projets
d’investissement y afférents contribuent non seulement à renforcer la compétitivité du
pays sur le plan international, mais aussi à améliorer significativement la qualité de vie
du citoyen marocain. À cet égard, le Gouvernement a entrepris une série d'actions
d'une portée décisive, dont l'une des plus notables est l'interconnexion stratégique
des bassins de Sebou et celui de Bouregreg, ayant permis le transfert de 186 millions
de mètres cubes d'eau. Cette initiative, hautement salutaire, a renforcé la résilience du
pays face aux cycles successifs de sécheresse, et permettra d'anticiper les défis futurs
en matière de gestion des ressources en eau.

Le secteur agricole, véritable pilier de l’économie nationale, bénéficie d’un soutien
renforcé, dans le cadre d’une approche intégrée visant à moderniser les pratiques et à
accroître la résilience face aux changements climatiques. Ainsi, le Gouvernement
poursuit la mise en œuvre de la stratégie « Green Génération 2020-2030 », qui
consiste à valoriser les acquis du secteur agricole et à renforcer sa résilience face aux
effets du changement climatique. En particulier, cette stratégie veille à
l’encouragement de l’irrigation d'appoint pour améliorer la productivité agricole et
accroître la production céréalière.

Parallèlement, le Gouvernement s’engage à mettre en œuvre les Orientations Royales
relatives au développement d’une industrie nationale dans le domaine du dessalement
de l’eau de mer. Ce projet ambitionne l’implantation d’entreprises marocaines
spécialisées dans la construction et l’entretien d’usines de dessalement. Aussi, l’accent
sera mis sur l’innovation et l’investissement dans les nouvelles technologies appliquées
à la gestion des ressources en eau, et ce en parallèle avec l’adoption d’un programme
plus ambitieux pour le traitement et la réutilisation des eaux usées.




     El
                                   NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


Quant au volet social de cette politique d’investissement, il se manifeste à travers une
attention soutenue à l'amélioration des services de la santé et de l'éducation, deux
leviers essentiels du développement humain. Les investissements dans ces secteurs
visent à accroître l'accès aux soins de santé et à l'éducation de qualité, notamment à
travers la mise en place et le renforcement des infrastructures sanitaires et le
développement continu de l’offre scolaire.
1.2 Renforcement de l'investissement public en                   tant   que catalyseur
    stratégique du développement Inclusif et durable
L’augmentation de l’enveloppe budgétaire allouée en 2024 aux investissements et
s’élevant à 335 milliards de dirhams, s’inscrit dans une dynamique d'intensification des
investissements publics. Particulièrement, et à la suite du tremblement de terre d’AI
Haouz ayant endommagé de nombreuses infrastructures dans plusieurs régions du
Royaume, une enveloppe prévisionnelle de 120 milliards de dirhams, étalée sur cinq
ans, a été accordée à la reconstruction des logements et des infrastructures touchées
par le séisme, ainsi qu’à la mise à niveau des régions sinistrées. En effet, ces travaux de
reconstruction post-séisme, transcendent une simple réponse conjoncturelle pour
s'inscrire dans une perspective stratégique de développement durable. Ainsi, chaque
projet de reconstruction programmé se déploie dans une logique de renforcement
structurel, visant à ancrer une croissance économique soutenue, qui assure à terme la
pérennité et l'épanouissement économique du pays.

Sur la même lancée, la nouvelle charte d'investissement incarne une volonté résolue
de promouvoir un développement inclusif et durable, tout en assurant une répartition
raisonnable entre les investissements publics et privés. Cette charte, adoptée dans le
cadre de la vision stratégique 2035, a pour objectif d’accroître la contribution des
investissements privés afin qu’ils atteignent les deux tiers du volume total des
investissements, tout en préservant une forte impulsion en faveur des investissements
publics.

Également, la nouvelle charte met un accent particulier sur la stimulation des
investissements privés, tant nationaux qu’étrangers. À cet égard, le Gouvernement
poursuit la mise en œuvre de la feuille de route stratégique 2023-2026 pour le
développement du climat des affaires, qui inclut des mesures incitatives substantielles.
En particulier, ces mesures englobent des incitations fiscales, l’amélioration du
système des marchés publics, la réduction des délais et la simplification des
procédures administratives. De même, elles visent à encourager les investisseurs à se
positionner dans des secteurs à haute valeur ajoutée, notamment les énergies
renouvelables, l’industrie agroalimentaire et les technologies innovantes. L’objectif
étant de favoriser non seulement la création d’emplois, mais aussi le transfert de
compétences et de technologies qui contribuent à renforcer la compétitivité de
l’économie nationale.
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


Un des axes majeurs de la charte d'investissement est de promouvoir une
gouvernance unifiée et décentralisée, d’où la volonté de stimuler une répartition
équitable des investissements sur l’ensemble du territoire national, avec une attention
particulière accordée aux régions les moins développées, et ce à l’effet de garantir
que les retombées économiques bénéficient à l’ensemble de la population. En
définitive, cette nouvelle charte s’inscrit dans une démarche cohérente de
développement inclusif et durable, mobilisant conjointement les ressources publiques
et privées, afin d’assurer une croissance équilibrée, durable et respectueuse des
exigences sociales et environnementales.

Pour optimiser l'impact de tels investissements, le Gouvernement ambitionne de
mettre en œuvre les diverses stratégies qui alignent les objectifs économiques avec
les exigences environnementales futures d’une part, et qui promeuvent l’inclusion
économique et sociale du citoyen marocain, d’autre part. Dans cette logique, les
projets d’investissement sont soumis à une évaluation rigoureuse, basée sur leur
avancement et leur adéquation avec les besoins sociaux et environnementaux. De
surcroît, des mécanismes de suivi permettront d'évaluer les résultats obtenus et
d'ajuster les stratégies en fonction des enseignements tirés.

En somme, l'investissement public constitue un catalyseur du premier plan pour la
réalisation d’un développement durable. En optimisant les stratégies d'investissement
et en tirant parti des opportunités d'amélioration, le pays vise non seulement à
renforcer son développement économique,               mais également à       préserver
l'environnement et à améliorer la qualité de vie des citoyens.
                                     NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



PARTIE II : Répartition régionale de l'investissement
public (Impact sur le développement des territoires et Plan
d’action pour 2025)
11.1     Aperçu sur Révolution           de     Penveloppe       budgétaire       accordée          à
        l’investissement public
       ► En termes de prévision

Au titre de l’année 2025, les efforts budgétaires du Gouvernement seront axés sur la poursuite
de l'élan donné à l'investissement public entre 2021 et 2024. Ainsi, l’enveloppe allouée au titre
du Projet de Loi de Finances 2025 devra atteindre 340 MMDH (voir graphique 1), soit une
progression de 1,5% (ou +5 MMDH) par rapport à son niveau en 2024, répartie entre le Budget
de l’Etat (Budget Général, Comptes Spéciaux du Trésor et Services de l’Etat Gérés de Manière
Autonome) d’un montant de 120,5 MMDH, les Etablissements et les Entreprises Publics d’un
montant de près de 138 MMDH, le Fonds Mohammed VI pour l’investissement d’un montant
de 45 MMDH, les Collectivités Territoriales d’une enveloppe de 21,5 MMDH et le Fonds spécial
pour la gestion des effets du tremblement de terre ayant touché le Royaume du Maroc d’un
montant de 15 MMDH.

             Graphique 1 : Evolution du volume global de l'investissement public
                            sur la période 2021-2025 (En MMDH)




       ► En termes de réalisation
Entre 2019 et 2023, l’exécution des crédits d'investissement au titre du Budget Général
témoigne d'une progression notable. En effet, le taux d'émission est passé de 77,9% en 2019 à
82,6 % en 2023 (voir graphique 2), enregistrant ainsi une augmentation de 4,73 points. Plus
précisément les émissions ont atteint un montant de 119,2 MMDH en 2023, contre 70,6 MMDH
en 2019, soit une hausse substantielle de 48,6 MMDH. Cette dynamique est principalement
attribuable à une politique volontariste de soutien à l'investissement public, conjuguée aux
efforts déployés pour accélérer la cadence d'exécution des projets, notamment à travers
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025


l'assainissement des crédits reportés dont le stock a été réduit, s'établissant à environ
12,97 MMDH en 2023 contre 15,75 MMDH en 2017, soit une baisse significative de 17,7% ou de
2,7 MMDH. De surcroît, le taux moyen des crédits reportés, par rapport aux crédits ouverts par
la Loi de Finances, s'est considérablement amélioré, en passant de 36 % en 2017 à 16 % en
2023.
    Graphique 2 : Evolution du taux d’émission des crédits d’investissement du Budget
                                  Général entre 2019 et 2023




11.2 La croissance régionale au service de l’intégration des territoires
Malgré les défis et crises structurelles que le Maroc a connus ces dernières années, notamment
sur les plans sanitaire, climatique et économique (inflation), les territoires ont su démontrer
une résilience significative, contribuant ainsi à renforcer la résilience au niveau national. Cette
capacité à surmonter les difficultés a été grandement soutenue par l'élan réformateur de la
régionalisation avancée, et les investissements importants et les projets ambitieux initiés par
les autorités publiques, tant au niveau national que territorial. Cela s'est traduit par une
modernisation et une diversification progressive des structures économiques locales, un
renforcement du tissu productif, ainsi qu’un accroissement des filets sociaux à l’échelle locale.
À cette échelle, l'investissement public joue un rôle central dans la mise en œuvre des
stratégies sectorielles et des grands projets d'infrastructure, conformément aux grandes
orientations du nouveau modèle de développement. Ce dernier met en avant l'ancrage
territorial, en tant que principal levier de la croissance économique. Dans cette optique, il est
essentiel de renforcer l’efficacité des investissements publics au niveau régional, afin d'assurer
une contribution optimale au développement durable, inclusif et harmonieux pour l’ensemble
des régions du Royaume.
    ► Dynamique de croissance en faveur d’une convergence régionale
Les initiatives publiques, notamment en matière d'amélioration des infrastructures et de la
mobilisation de l'investissement public à l'échelle régionale, ont joué un rôle clé dans le
renforcement du développement des territoires. Elles ont, également, contribué à stimuler les
investissements privés locaux, moteurs essentiels de la croissance et de l'amélioration de la
qualité de vie des citoyens. Ainsi, ces dernières années, des dynamiques remarquables de
croissance territoriales ont été observées, portées essentiellement par les régions en dehors
de la dorsale Tanger-EI Jadida. Toutefois, malgré ces avancées, les inégalités régionales
persistent et continuent d'impacter la structure de l'activité économique.
                                     NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


Ainsi, l’analyse de la structure régionale du PIB nominal, montre que la région de Casablanca-
Settat a réalisé la part moyenne la plus importante du PIB, aux prix courants, durant la période
2014-2022, soit une contribution au PIB de l’ordre de 32,1%, suivie par les régions de Rabat-
Salé-Kénitra (16%) et de Tanger-Tétouan-AI Hoceima (10,5%). Ces trois régions totalisent
58,6% du PIB national, ce qui souligne l'importance de consacrer plus d’efforts en faveur d’une
meilleure inclusion spatiale.
    Carte : PIB par habitant et contribution régionale au PIB sur la période 2014-2022




             Source : MEF

En matière de dynamique de croissance nominale, il convient de souligner l’émergence
remarquable des régions du Sud. En effet, la région de Laâyoune - Sakia El Hamra a enregistré
un rythme de croissance annuel moyen de 9,1%, suivie des régions de Guelmim-Oued Noun
(5,5%), de Dakhla-Oued Ed-Dahab (5,4%) et de Drâa-Tafilalet (5,2%). Dans le même élan, la
région de Tanger-Tétouan-AI Hoceima a affiché un taux de croissance de 5%, dépassant ainsi
la moyenne nationale (3,6%). Cette performance est le fruit des investissements majeurs
réalisés dans la région, ayant transformé en profondeur sa structure productive, consolidant
ainsi son rôle en tant que deuxième plateforme industrielle et commerciale du Royaume.
Parallèlement, la région de Rabat-Salé-Kénitra a enregistré un taux de croissance de l’ordre de
3,8%, dépassant ainsi le niveau national, alors que la région de Casablanca-Settat a affiché un
taux équivalent au niveau national (3,6%) sur la période 2014-2022. Ces dynamiques récentes
pourraient contribuer à résorber les inégalités territoriales en matière de répartition de la
richesse nationale.
Concernant le PIB nominal par habitant, 5 régions sur 12 ont un niveau supérieur à la moyenne
nationale (33.172 dh/habitant) sur la période 2014-2022. Il s’agit de Dakhla-Oued Ed-Dahab
(83.074), de Laâyoune-Sakia El Hamra (53.776), de Casablanca-Settat (51.958), de Rabat-Salé-
Kénitra (39.169) et de Guelmim-Oued Noun (37.993).
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025


                       Graphique 3 : PIB/habitant par région (2014-2022)
               90000
          c    80000
          £    70000
          o    60000
          “    50000
          ra   40000
         -E    30000
         Q     20000
         £     10000
                   0




        Source : MEF

En termes de rythme de croissance du PIB par habitant, la région de Laâyoune-Sakia El Hamra
s'est particulièrement distinguée avec la plus forte progression, enregistrant un taux annuel
moyen de 7,6% au cours de la période 2014-2022. Elle est suivie par la région de Guelmim-
Oued Noun enregistrant une croissance de 5%, puis Drâa-Tafilalet avec 4,6%, puis Tanger-
Tétouan-AI Hoceima avec 3%, Rabat-Salé-Kénitra avec 2,8% et Béni Mellal-Khénifra avec 2,7%.
Par ailleurs, la région de l’Oriental a enregistré un rythme de croissance du PIB par habitant
similaire à la moyenne nationale (2,6%). En revanche, la région de Casablanca-Settat a
enregistré une croissance de 2,3%, suivie de la région de Souss-Massa avec 2,2%, de Dakhla-
Oued Ed-Dahab avec 1,5%, de Marrakech-Safi avec 1,4% et de Fès-Meknès avec 1,2%. Ces
dynamiques contrastées révèlent un processus de convergence régionale, caractérisé par une
croissance plus vigoureuse dans les régions à faible contribution au PIB, tandis que les pôles
économiques traditionnels semblent connaître un certain ralentissement.
    ► Configuration économique territoriale : une diversité sectorielle
L’analyse de la répartition sectorielle des valeurs ajoutées régionales, aux prix courants, fait
ressortir que trois régions se distinguent par leur contribution majeure à la valeur ajoutée des
trois grands secteurs d’activité, en l’occurrence Casablanca-Settat, Rabat-Salé-Kénitra et
Tanger-Tétouan-AI Hoceima.
En effet, la région de Rabat-Salé-Kénitra a enregistré la part moyenne la plus importante de la
valeur ajoutée primaire durant la période 2014-2022, soit 16,5% de la valeur ajoutée nationale
du secteur. La région de Fès-Meknès se positionne au deuxième rang avec une moyenne de
15,5%, suivie des régions de Casablanca-Settat (12,7%), de Marrakech-Safi (11,4%), de Souss-
Massa (9,9%), de Tanger-Tétouan-AI Hoceima (9,4%) et de Béni Mellal-Khénifra (9,3%). Ces
sept régions totalisent 84,7% de la valeur ajoutée nationale du secteur primaire, aux prix
courants.
S’agissant des activités secondaires, la région de Casablanca-Settat a réalisé la part moyenne
la plus importante de la valeur ajoutée secondaire nationale aux prix courants, atteignant
45,4% durant la période considérée. La région de Tanger-Tétouan-AI Hoceima se positionne à
la deuxième place, à hauteur de 13,3%, suivie des régions de Rabat-Salé-Kénitra (9,6%) et de
Béni Mellal-Khénifra (6,7%). Parallèlement, la région de Casablanca-Settat a affiché la
contribution moyenne la plus importante du secteur tertiaire au niveau national (29,9%)
durant la période 2014-2022, suivie des régions de Rabat-Salé-Kénitra (19,3%), de Marrakech-
Safi (9,2%), de Tanger-Tétouan-AI Hoceima (9,1%) et de Fès-Meknès (8,3%).
                                      NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



11.3 L’intervention de l’Etat en termes d’investissement
11.3.1 Renforcement des fondements de l’Etat social
11.3.1.1 Éducation nationale et enseignement préscolaire
Conscient de l’importance du secteur de l’éducation nationale et de l’enseignement
préscolaire dans l’édification de l’Etat social, et conformément aux Hautes Orientations
Royales relatives à la réforme du système éducatif, le Gouvernement continue à mettre en
œuvre les dispositions de la loi-cadre n°51-17 relative au système d'éducation, de formation et
de recherche scientifique, ainsi que les programmes de la « Feuille de route 2022-2026 pour
une école publique de qualité », notamment en ce qui concerne :
   •   La mise en œuvre du programme national d'expansion et de développement de
       l'enseignement préscolaire, pour consacrer l'équité et l'égalité des chances pour la
       tranche d’âge de 4 et 5 ans, dans la perspective de sa généralisation à l’horizon 2028 ;
   •   La réhabilitation et l’entretien des établissements et équipements existants, ainsi que la
       construction de nouveaux établissements dans les zones rurales et semi-urbaines.
La planification stratégique dans le secteur accorde une importance primordiale à l'expansion
de l'offre scolaire. Dans ce sens, et afin de répondre aux besoins croissants en matière
d'infrastructures éducatives, 189 établissements (dont 68% en milieu rural) ont été créés pour
la rentrée scolaire 2024-2025, contre 237 établissements et 2.313 salles de classe par
extension ouverts lors de la rentrée scolaire 2023-2024. À ces réalisations s'ajoutent
242 établissements en construction et 136 autres en phase de lancement. La répartition des
projets par région est présentée dans le tableau ci-dessous :

                                         Rentrée scolaire 2024-2025                     Nombre
                                                                                  d’établissements en
             Région                      Nombre
                                                               Dont            cours de construction au
                                    d’établissements
                                                         (écoles satellites)     titre de l’année 2024
                                    scolaires réalisés
 Tanger-Tétouan-AI Hoceima                  30                    9                      19
 L’Oriental                                 23                    4                       5
 Fès-Meknès                                 21                    3                      50
 Rabat-Salé-Kénitra                         24                    1                      21
 Béni Mellal-Khénifra                       14                    6                      31
 Casablanca-Settat                          24                    -                      19
 Marrakech-Safi                             16                    -                       8
 Drâa-Tafilalet                             13                    7                      19
 Souss-Massa                                19                    -                      52
 Guelmim-Oued Noun                           3                    1                       3
 Laâyoune-Sakia El Hamra                     -                    -                      14
 Dakhla-Oued Ed-Dahab                        2                    -                       1
              Total                        189                   31                      242

Grâce à ces investissements massifs en matière d'infrastructures scolaires, les établissements
scolaires à travers le Royaume ont pu accueillir, au titre de la rentrée scolaire 2024-2025, près
de 8 millions d’élèves (6,9 millions d’élèves dans les établissements publics et 1,1 million
d’élèves dans les établissements privés) répartis sur les trois cycles, confirmant ainsi la volonté
gouvernementale de garantir à tous les élèves un environnement d'apprentissage adapté à
leurs besoins, et de lutter contre la surcharge des classes.
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


Par ailleurs, et dans une logique de cohérence avec les objectifs fixés, le programme
d'investissement de l’année 2025 accordera une priorité particulière à :
    ►   La généralisation et le développement de l’enseignement préscolaire de qualité :
Il s’agit d’intensifier les efforts pour atteindre un taux de scolarisation supérieur à 80% en
2025, notamment à travers la construction et l’équipement de 3.200 nouvelles salles de classe
et la réhabilitation et l'équipement de 640 salles existantes.
    ►   La construction de nouveaux établissements scolaires :
Pour satisfaire la demande croissante en matière d’éducation, il est question de procéder au
titre de l’année 2025 à la construction de 181 nouveaux établissements scolaires, et 2.094
salles de classe dans le cadre de l'extension des établissements existants, selon la répartition
régionale présentée dans le tableau ci-après, et ce pour une enveloppe budgétaire d'environ
2,3 MMDH :
                                                                                      Nombre annuel de
                                        Nombre annuel d’établissements
                                                                                       salles de classe
              Région                                                                  par extension des
                                 Écoles
                             communautaires
                                              Primaires   Collèges   Lycées   Total    établissements
                                                                                           existants
 Tanger-Tétouan-AI
                                    3            8           12        7       30           286
 Hoceima
 L’Oriental                         0            5           5         4       14            198

 Fès-Meknès                         2            4           5         4       15           276

 Rabat-Salé-Kénitra                 1            8           8         6       23           245

 Béni Mellal-Khénifra               3            4           2         1       10            221

 Casablanca-Settat                  4            11          9         7       31            231

 Marrakech-Safi                     1            3           11        8       23            154

 Drâa-Tafilalet                     3            3           6         2       14           235

 Souss-Massa                        3             7          3         3       16            218

 Guelmim-Oued Noun                  0            0           0         1        1            20

 Laâyoune-Sakia El Hamra            0             1          1         1        3            10

 Dakhla-Oued Ed-dahab               0             1          0         0        1             0

              Total                20            55         62        44       181         2.094


Dans le même sillage, l’offre éducative sera renforcée par la construction de 15 nouveaux
internats et la transformation de 40 écoles satellites en écoles communautaires accueillant
déjà plus de 300 élèves.
                                              NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


    ► La réhabilitation des établissements d'éducation et de formation existants :
Ces opérations bénéficiant d'un budget d'environ 2,1 MMDH en 2025, prévoient le
remplacement des salles construites en préfabriqué, la réhabilitation de tous les espaces
éducatifs en les raccordant aux réseaux d’eau, d'électricité et d’assainissement, ainsi que la
construction des murs de clôture et la mise en place d’un système de maintenance préventive
garantissant la préservation de ces espaces en bon état.
Le nombre d’opérations prévues en 2025 par catégorie, est illustré dans le tableau ci-dessous :

                                        Opération                                  Nombre en 2025

Travaux de réhabilitation et de réparation des établissements pionniers :              2.500
  Primaire                                                                             2.000
  Collégial                                                                             500
Travaux de réhabilitation et de réparation des autres établissements :                 7.400
  Clôtures et mûrs                                                                      1.000
  Raccordement et approvisionnement en eau potable                                      1.200
  Renouvellement et réhabilitation du réseau                                            800
  Raccordement et approvisionnement en électricité                                      800
  Construction et réparation d’installations sanitaires                                2.000
  Raccordement au réseau d’assainissement sanitaire                                     800
  Remplacement des classes construites par des installations démontables                600
  Travaux de réhabilitation et de réparation du secondaire                              200
                                           Total                                       9.900

11.3.1.2 Enseignement supérieur, recherche scientifique et innovation
Dans le cadre du Plan d’Accélération de la Transformation de l’Ecosystème de l’Enseignement
Supérieur, de la Recherche Scientifique et de l’innovation (PACTE ESRI 2030), s’inspirant des
Orientations Royales pour le secteur, de la Vision Stratégique pour la Réforme 2015-2030, de
la Loi-Cadre n°51-17 et des recommandations du Nouveau Modèle de Développement, le
Gouvernement intensifie ses efforts pour la réalisation des projets de construction,
d'équipement et de réhabilitation des établissements universitaires à l’échelle nationale, afin
de répondre au besoin évolutif d’un enseignement supérieur de haute qualité. Ces projets
s’inscrivent dans une dynamique de résolution des problèmes auxquels est confronté le
système d'enseignement supérieur au Maroc, avec un accent particulier sur :

      ■ L’accompagnement de l’accroissement de la demande, en assurant un équilibre dans
        la distribution de l'offre d’enseignement supérieur entre les régions ;
      ■ La création de nouveaux modèles d'établissements universitaires à accès ouvert,
        ciblant des domaines de formation spécifiques (Facultés d'économie et de gestion,
        Facultés de langues et des arts...) ;
      ■ L’appui à la formation dans le domaine médical qui s’inscrit dans le cadre du
        programme « Renforcement de la densité du personnel de la santé à l’horizon 2030 »,
        en relation avec le chantier Royal de généralisation de la couverture médicale ;
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


      ■ Le soutien aux filières de formation en éducation dans les universités pour répondre
        aux besoins en enseignants, dans le cadre du programme de « Formation des
        enseignants des cycles primaire et secondaire à l’horizon 2025 » ;
      ■ L’augmentation du nombre des nouveaux inscrits dans les établissements
        universitaires à accès régulé en développant l'offre y afférente.
    ► Établissements d’enseignement supérieur dont la construction est achevée en
      2024 :
Le réseau des établissements d’enseignement supérieur s'est étoffé en 2024, grâce à
l’achèvement de la construction de cinq (5) nouveaux établissements :

                                                                                                      Coût
             Région                                          Projet
                                                                                                     (MDH)
                                 Construction et équipement de la Faculté Polydisciplinaire à
                                                                                                       130
                                 Ksar El Kbir
 Tanger-Tétouan-AI Hoceima
                                 Construction et équipement de l'École Supérieure de
                                                                                                       81
                                 Technologie de Tétouan

                                 Construction et équipement de l'Ecole Nationale de
                                                                                                       60
                                 Commerce et de Gestion de Béni Mellal

 Béni Mellal-Khénifra            Construction et équipement de l'Ecole Supérieure de
                                                                                                       80
                                 Technologie à Fquih Ben Saleh

                                 Construction et équipement de la Faculté de l’Economie et de
                                                                                                       70
                                 Gestion à Béni Mellal
                                              Total                                                    421


    ► Établissements d’enseignement supérieur en cours de construction :
Le tableau ci-dessous présente la répartition régionale des projets de construction
d'établissements d'enseignement supérieur en cours, mettant en évidence l'investissement
accordé par le Royaume au développement d'un réseau universitaire étendu et diversifié :

                                                                                                     Taux de
                                                                            Coût       CP 2025     réalisation
      Région                                Projet
                                                                           (MDH)        (MDH)    à fin juin 2024
                                                                                                      (en %)

Tanger-Tétouan-AI       Construction et équipement du Campus
                                                                             300         50            85
Hoceima                 Universitaire d’AI-Hoceima

                        Construction et équipement de l’Ecole
                                                                             151         20            55
                        Supérieure d’Education et de Formation d’Oujda
L'Oriental
                        Création de la maison d’Afrique et extension et
                        aménagement du complexe sportif de l’université      140         30            10
                        Mohamed Premier Oujda
                        Construction et équipement de l’Ecole
Rabat-Salé-Kénitra      Supérieure d’Education et de Formation à             157         20            10
                        Kénitra
                        Construction et équipement de l’Ecole
Béni Mellal-
                        Supérieure d’Education et de Formation à Béni        93          10            90
Khénifra
                        Mellal
                                          NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                    Construction et équipement de l’Ecole Nationale
                                                                         95    10              98
                    des Sciences Appliquées à Béni Mellal
Béni Mellal-
Khénifra
                    Construction et équipement de la Faculté de
                                                                        445    80              5
                    Médecine et de Pharmacie de Béni Mellal

                    Construction et équipement de l'Ecole Nationale
                                                                         70    20              10
                    de Commerce et de Gestion d’EI Jadida
                    Construction et équipement de l’Ecole
                    Supérieure d’Education et de Formation d’EI          105   40              10
                    Jadida
                    Construction et équipement de l'institut des
                                                                         150   10              75
                    Sciences de Sport de Settat
Casablanca-Settat
                    Construction et équipement de l’Ecole
                    Supérieure d’Education et de Formation de            68     3              38
                    Berrechid
                    Construction et équipement de la Faculté des
                                                                        100    20         Phase d’étude
                    Langues, Arts et Sciences Humaines à Settat

                    Construction et équipement de la faculté
                                                                        100    20         Phase d’étude
                    d’économie et de gestion à Settat

                    Construction et équipement du nouveau siège de
Marrakech-Safi                                                           60    20              10
                    l'ENCG de Marrakech

                    Construction et équipement de l'Ecole
                                                                         70    20         Phase d'étude
                    Supérieure de Technologie (EST) d’Ouarzazate
Drâa-Tafilalet
                    Construction et équipement de la Faculté de
                                                                        445    80              5
                    Médecine et de Pharmacie d’Errachidia

                    Construction et équipement de l'Ecole
Souss-Massa                                                              94    28              80
                    Supérieure d’Education et de Formation d'Agadir

Guelmim-Oued        Construction et équipement de la Faculté de
                                                                         315   64         Phase d’étude
Noun                Médecine et de Pharmacie de Guelmim

Laâyoune-Sakia El   Construction et équipement de la Faculté de
                                                                        468    50              95
Hamra               Médecine et de Pharmacie de Laâyoune

                               Total                                   3.426   595              -


    ► Lancement de nouveaux projets de construction :
L'année 2025 sera marquée par le lancement d'un plan d'investissement ambitieux, visant à
doter le Royaume de nouvelles infrastructures universitaires, réparties par région comme suit :

                                                                                Coût         CP 2025
        Région                                   Projet
                                                                               (MDH)          (MDH)

                        Construction et équipement de l'Ecole Nationale de
 Fès-Meknès                                                                         70          20
                        Commerce et de Gestion de Meknès


                        Construction et équipement de l'Ecole Nationale de
 Drâa-Tafilalet                                                                     100         20
                        Commerce et de Gestion d’Ouarzazate
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025


                        Construction et équipement des Instituts Thématiques
                                                                                       150      50
                        de Recherche
                        Construction, aménagement et équipement des espaces
 Multirégional                                                                        176,9     46
                        Code 212
                        Construction, aménagement et équipement des Centres
                                                                                       60       20
                        d'excellence
                                      Total                                           556,9    156

Par ailleurs, et dans le cadre du développement et de l’amélioration des services sociaux
destinés aux étudiants, notamment les plus démunis, le département de l’enseignement
supérieur poursuivra en 2025 les travaux de construction et d’équipement des cités
universitaires, pour un coût global de 220 MDH,

11.3.1.3 Formation professionnelle
En s'inscrivant dans la dynamique impulsée par la feuille de route présentée devant Sa Majesté
le Roi, que Dieu L’assiste, le 4 avril 2019, le Gouvernement poursuit la mise en œuvre d’une
panoplie de mesures visant à moderniser le système de formation professionnelle, et à
garantir une offre actualisée qui répond aux besoins des professionnels locaux et nationaux,
tous secteurs confondus.
Cette modernisation s'est concrétisée par la mise en place d’une nouvelle génération
d’instituts de formation professionnelle, caractérisée par une implication plus élargie des
associations et des organisations professionnelles, tout en leur attribuant un rôle
prépondérant dans la gestion du dispositif de formation professionnelle, et ce dans un cadre
de partenariat public privé.
C’est ainsi que, (10) instituts spécialisés ont été mis en place dans les secteurs de l’industrie
automobile (03), l’aéronautique (01), le textile et l’habillement (02), les énergies renouvelables
et l’efficacité énergétique (03) et le transport et la logistique (01).
L'engagement demeurera fort en 2025, pour la poursuite de la concrétisation d'une série
d'établissements de formation, en collaboration avec la Confédération Générale des
Entreprises du Maroc (CGEM), la Fédération Marocaine de l'industrie et de l'innovation
Pharmaceutiques et les conseils de régions concernées. Il s’agit en effet de :
    ► La poursuite de la construction des établissements de formation :
                                                                                       Coût   CP 2025
        Région                                    Projets
                                                                                      (MDH)    (MDH)
                        Institut de formation à l’entrepreneuriat et au middle
                                                                                       77       15
                        management à Casablanca
                        Institut de formation dans les métiers de l’industrie et de
                                                                                       130      35
                        l’innovation pharmaceutique à Casablanca
 Casablanca-Settat      Hub centre national de la formation des formateurs et
                        des tuteurs dans les métiers de l’automobile à                 85       25
                        Casablanca
                        Internat du centre de formation dans les métiers du
                                                                                       38       13
                        transport et de la logistique à Casablanca
                        Institut de formation dans les métiers de l’eau et de
 Fès-Meknès                                                                             61      50
                        l’assainissement et l’environnement à Fès
                                     Total                                             391      138
                                             NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


Par ailleurs, il convient de souligner que la dynamique initiée pour la concrétisation du
programme des Cités des Métiers et des Compétences (CMC), colonne vertébrale de la
nouvelle feuille de route pour le développement de la formation professionnelle, se poursuivra
en 2025, comme en atteste le tableau ci-dessous :

                                    Projet                                Etat d’avancement

 CMC région Souss-Massa
 CMC région Laâyoune-Sakia El Hamra                                           Achevées
                                                                   Mise en service effective pour la
 CMC région Rabat-Salé-Kénitra                                       rentrée scolaire 2023-2024
 CMC région de L’Oriental

 CMC région Tanger-Tétouan-AI Hoceima
 CMC région Béni Mellal-Khénifra                                               Achevées
                                                                  Mise en service au cours de l’année
 CMC région Casablanca-Settat                                             scolaire 2024-2025

 CMC région Dakhla-Oued Ed-Dahab*

 CMC région Marrakech-Safi
                                                                               En cours
 CMC région Guelmim-Oued Noun                                               d’achèvement
 CMC région Drâa-Tafilalet                                      Mise en service prévue pour la rentrée
                                                                         scolaire 2025-2026
 CMC région Fès-Meknès
(*) En phase finale d'équipement.


11.3.1.4 Santé et protection sociale
Sous l'impulsion des Hautes Instructions de Sa Majesté le Roi Mohammed VI, que Dieu
L’assiste, visant une réforme en profondeur du système national de santé, et dans le souci de
répondre aux enjeux d'un contexte socio-économique en pleine mutation, le Maroc s'est
engagé dans une transition majeure vers un modèle de développement centré sur le social. La
généralisation de l'Assurance Maladie Obligatoire de base à tous les citoyennes et citoyens, y
compris ceux n’ayant pas la capacité de s’acquitter des droits de cotisation, traduit une
ambition claire : bâtir un État social solide, où les inégalités en matière d’accès aux soins sont
réduites.
En vue d’accompagner le chantier Royal de la généralisation de la couverture médicale, le
Gouvernement poursuit la refonte globale du système national de santé, qui s’articule autour
des axes suivants : la mise à niveau de l’offre de soins, la valorisation et le renforcement des
ressources humaines, le renforcement de la gouvernance du système national de santé et la
mise en place d'un système d'information intégré.
Ainsi, et dans le but de consolider l'offre de soins et d'optimiser la qualité des services de
santé, l'année 2025 sera marquée par la poursuite des efforts du Gouvernement pour mettre
en place une panoplie de projets d’investissement à l'échelle nationale, qui se présentent
comme suit :
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025


    ► La création des CHU dans chaque région pour le renforcement du maillage
      hospitalier universitaire :
Il s’agit de la poursuite de la construction et d’équipement des CHU de Rabat, de Laâyoune,
de Guelmim, de Béni Mellal et d’Errachidia :
                                                                                                      Etat d’avancement à fin
                                                     Capacité           Coût          CP 2025                juin 2024
          Région                   Projet                                                                               Taux de
                                                      litière          (MDH)           (MDH)          Emissions
                                                                                                                       réalisation
                                                                                                       (MDH)
                                                                                                                         (en %)
                             CHU Ibn Sina de
 Rabat-Salé-Kénitra                                    1.044           6.530           1.400           1.157,35            28
                             Rabat
                                                                                                                     Lancement des
 Béni Mellal-Khénifra        CHU de Béni-Mellal            500         2.400           134,82             —          études en cours

 Souss-Massa                 CHU d’Agadir                  867         2.375            300            1.951,21            98

 Drâa-Tafilalet              CHU d’Errachidia              500         1.800           239,89           0,10         Phase d’étude

                                                                                                                     Préparation du
 Guelmim-Oued Noun           CHU de Guelmim                300          800             65                —          lancement des
                                                                                                                         études
 Laâyoune-Sakia El Hamra     CHU de Laâyoune               500         1.800           160,66          685,86              60

                           Total                                       15.705     2.300,37            3.794,52

    ► Le renforcement du réseau hospitalier régional :
L’année 2024 a connu l’inauguration du Centre Hospitalier Zemmouri de Kénitra, du Centre
Hospitalier Mohammed VI dans la province d’AI Hoceima et des hôpitaux de proximité à
Figuig, Talsint et Ahfir. La dynamique d'extension et de mise à niveau de l’offre hospitalière à
l'échelle régionale se maintient en 2025, notamment par :
       > La poursuite des travaux de construction/reconstruction et d’équipement des
         établissements et centres hospitaliers :
                                                                                                        Etat d’avancement à fin
                                                                                             CP                juin 2024
                                                             Capacité          Coût
              Région                    Projet                                              2025                         Taux de
                                                              litière         (MDH)                      Emissions
                                                                                           (MDH)                        réalisation
                                                                                                          (MDH)
                                                                                                                          (en %)
                               Reconstruction du CHP
                                                                 120            488           65,27        127,71           54
                               d’Ouazzane
 Tanger-Tétouan-AI Hoceima     Construction de
                               l’hôpital des spécialités         380            680          236,29       340,94            85
                               de Tétouan
                               Reconstruction du CHR                                                           -
                                                                 250            913           20,38                    Phase d'étude
                               d'Oujda
                               Reconstruction du CHP                                         153,79        26,47
                                                                 175            480                                         60
                               de Berkane
                               Reconstruction du CHP
 L’Oriental                    de Nador
                                                                 250            536          218,16        152,32           46

                               Reconstruction du CHP                                                          0,49
                                                                 160            415          59,46                     Phase d’étude
                               de Taourirt
                               Reconstruction du CHP
                                                                 190            420            7,88           3,74     Phase d'étude
                               de Guercif
                               Construction du CHP de                                                                   Travaux en
 Fès-Meknès                    Moulay Yaacoub
                                                                 150            571           10,00
                                                                                                                           cours
                                        NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


                                                                                     Etat d’avancement à fin
                                                                            CP              juin 2024
                                                      Capacité    Coût
           Région                  Projet                                  2025                    Taux de
                                                       litière   (MDH)               Emissions
                                                                          (MDH)                   réalisation
                                                                                      (MDH)
                                                                                                    (en %)
                          Construction de
                          l'hôpital des Spécialités                                               Travaux en
Fès-Meknès                                              120       427      10,00
                          à la préfecture de                                                         cours
                          Meknès
                          Reconstruction du CHP                                                   Travaux en
Rabat-Salé-Kénitra        de Khémisset
                                                        260       670      14,66         -
                                                                                                     cours

                          Reconstruction du CHR                            42,94       6,49
                                                        450       900                                 10
                          de Béni Mellal
                          Construction du CHP de                                                  Travaux en
Béni Mellal-Khénifra      Fquih Ben Salah
                                                        175       460       71,81
                                                                                                     cours

                          Reconstruction du CHP                                          -
                                                                                                  Travaux en
                                                        120       370      17,99
                          d'Azilal                                                                   cours

                          Reconstruction du CHP                                                   Lancement
                                                        250       505      23,00
                          de Mohammadia                                                           des études
Casablanca-Settat         Reconstruction de
                          l’Hôpital Psychiatrique       300       250      16,00                 Phase d’étude
                          à Berrechid
                          Construction du CHR à                                          -
                                                        250       800       7,00                 Phase d'étude
                          Tamansourt
                          Reconstruction du CHP                                                   Travaux en
Marrakech-Safi                                          120       604       2,54        0,81
                          d’Er-Rhamna                                                                cours
                          Construction de
                          l’hôpital de Proximité         45       84       33,16       21,84          60
                          de Tamanar
                          Reconstruction du CHP
                                                        120       330      88,96      143,45          90
                          de Tinghir
Drâa-Tafilalet            Construction de
                          l’hôpital des spécialités     145       570       3,71       0,24      Phase d’étude
                          d’Ouarzazate
                          Construction de
                          l’hôpital de proximité de      80       150       11,00        -       Phase d'étude
                          Taliouine
                                                                                                 Marché « GO-
                          Construction de
                                                                                                  Etanchéité »
Souss-Massa               l’hôpital de Proximité         45       158      26,72        0,31
                                                                                                    en cours
                          d’Ouled Berhil                                                         d'approbation
                          Construction de
                                                                                                  Travaux en
                          l’hôpital de proximité         45       145      34,84       0,38
                                                                                                     cours
                          de Tighirt
                          Reconstruction du CHR
                                                        250       700      55,00       131,72         62
                          de Guelmim
Guelmim-Oued Noun
                          Reconstruction du CHP                                                   Travaux en
                                                        120       540       2,61       45,28
                          de Sidi Ifni                                                               cours

                          Construction du CHP de
Laâyoune-Sakia El Hamra   Tarfaya
                                                         70       140       3,23       91,94          99

                          Construction du CHR de
                                                        450       600       5,00                 Phase d’étude
                          Dakhla
Dakhla-Oued Ed-Dahab      Construction de
                          l’hôpital de Proximité de      45       220      10,95                 Phase d’étude
                          Bir Gandouz
                          Total                                  13.126   1.252,35   1.094,13
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025


      > Le lancement de nouveaux projets :
                                                                         Capacité       Coût      CP 2025
          Région                               Projet
                                                                          litière      (MDH)       (MDH)
                            Reconstruction du CHP de Sefrou                 120          500         75
 Fès-Meknès
                            Reconstruction du CHP de Taounate               120          547        116

                            Construction du Centre de
 Marrakech-Safi                                                             160          187,7       24
                            Traumatologie à Tamansourt

                                       Total                                           1.234,70     215

 >   La poursuite de la mise en œuvre du programme de réhabilitation de près de 1.400
     établissements de soins de santé primaire au niveau des 12 régions. À rappeler que ces
     établissements, dont les deux tiers sont situés dans des zones rurales et reculées,
     constituent la première destination des patients et jouent un rôle crucial dans la
     proximité des services de santé pour les citoyens, ainsi que dans la garantie de la qualité
     des soins. À ce jour, le Gouvernement a réussi à réhabiliter un total de 872 centres de
     santé, auxquels s’ajoutent 524 centres (3eme tranche) devant être achevés avant
     fin avril 2025.

11.3.1.5 Culture
Le nouveau modèle de développement a mis l'accent sur la diversité culturelle comme vecteur
d'ouverture, de dialogue et de cohésion sociale. Dans cette même lignée, le programme
gouvernemental a placé la promotion et le développement des affaires culturelles au cœur de
ses priorités. C’est ainsi que le Gouvernement ambitionne de faire de la politique culturelle et
artistique, un fondement central du renforcement de l'identité nationale et de l'ouverture aux
autres cultures, et ce en adoptant une approche participative et intégrée.
Dans ce sens, l’année 2025 sera caractérisée par le lancement de nouveaux projets culturels,
dont la répartition régionale est présentée dans le tableau suivant :

                                                                                        Coût      CP 2025
       Région                                       Projet
                                                                                       (MDH)       (MDH)

                      Restauration de la muraille historique et réhabilitation de
                                                                                          30         20
                      l'ancienne médina de Taza
Fès-Meknès
                      Construction d'un complexe culturel (2ème tranche) dans la
                                                                                          14         14
                      ville de Taza
                      Réalisation du Musée national de l'archéologie et des sciences
                                                                                         100         50
                      de la terre à Rabat

                      Amélioration du paysage urbain de la ville de Rabat                100         30
Rabat-Salé-Kénitra
                      Mise en valeur patrimoniale de la ville de Rabat                    30         30


                      Réhabilitation de Hay Al Habous à Yaacoub El Mansour                9,2       9,2


                                       Total                                           283,20      153,20
                                           NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


De même, il sera question en 2025 de poursuivre la réalisation des projets culturels ci-après :

                                                                                          Coût      CP 2025
       Région                                       Projet
                                                                                         (MDH)       (MDH)
Tanger-Tétouan-AI      Réhabilitation et valorisation de la Médina de Tanger 2020­
                                                                                           850         15
Hoceima                2024
                       Renforcement des infrastructures culturelles et diversification
L'Oriental                                                                                 380         40
                       de l'offre culturelle dans la région de l’Oriental

                       Réalisation du pôle culturel à Fès                                 189,38      15,34

                       Construction et équipement de cinq centres de proximité et de
                                                                                            30        6,75
                       valorisation de la culture montagnarde à Boulemane

Fès-Meknès             Construction d’un centre culturel à Galdamane à Taza                 14         5


                       Construction d’un centre culturel à Moulay Yaacoub                   14         3


                       Construction d’un centre culturel à Kissane                          7           1

                       Construction d'un centre culturel au pôle urbain de Bouknadel,
                                                                                            22         8
                       Province de Salé

Rabat-Salé-Kénitra     Création de la Foire de Rabat                                        75         65


                       Restauration de la Kasbah de Témara                                 40          20

                       Réhabilitation et aménagement de la Kasbah Ismailia (Kasbat                      7
                                                                                            10
                       Tadla)
Béni Mellal-Khénifra
                       Restauration et mise en valeur de la Kasbah Zidaniya                13,2         7

                       Restructuration de la station sportive et récréative de
                                                                                           105         5
                       l’Oukaimden
Marrakech-Safi
                       Construction du complexe Cité des Arts à la ville d'Essaouira        60         15


Drâa-Tafilalet         Restauration et mise en valeur du site de Sijlmassa à Rissani       246       141,14


                       Construction et équipement du Grand Théâtre d'Agadir                250         4


Souss-Massa            Aménagement et équipement de la Maison des Arts à Agadir            120         15

                       Développement des secteurs de la culture, la jeunesse et du
                                                                                          32,06       2,06
                       sport pour la ville d'Agadir
                                        Total                                            2.457,64    375,29


11.3.1.6 Jeunesse
Les perspectives du développement du Maroc et la réussite des changements des politiques
socio-économiques dépendent, entre autres, des conditions d’implication des jeunes. Dans ce
sens, le Gouvernement, en collaboration avec les acteurs concernés, a élaboré une « Stratégie
Nationale de la Jeunesse ». Elle vise à promouvoir l'engagement et l'autonomisation des
jeunes dans divers domaines, tels que l'éducation, l'emploi, la santé et la participation
citoyenne.
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


C’est dans le cadre de la mise en œuvre de cette stratégie que s’inscrivent les principaux
projets d’investissement, nouveaux et en cours de réalisation, au titre de l’année 2024, dont la
répartition régionale pour l’année 2025 est donnée comme suit :

                                                                                           Coût     CP 2025
       Région                                    Projet
                                                                                          (MDH)      (MDH)
 Tanger-Tétouan-AI
                        Aménagement de la colonie de vacances Ajdir                         13            10
 Hoceima

                        Construction de la colonie de Vacances Kharzouza                   88             30

 Fès-Meknès             Construction de la colonie de Vacances Toumliline                  33,5           10

                        Programme de la promotion de la région Fès-Meknès
                                                                                           42,9           20
                        2020-2023
                        Création de la grande plateforme des Jeunes à la
 Rabat-Salé-Kénitra                                                                        200            80
                        préfecture de Rabat

                        Construction de la colonie de Vacances Oualidia                     91            30
 Casablanca-Settat
                        Programme de Développement Régional Casablanca-
                                                                                           150            30
                        Settat

 Marrakech-Safi         Construction de la colonie de vacances Sidi Fares                  70             30

                                     Total                                                688,4           240

11.3.1.7 Habous et affaires islamiques
Le plan d’action du Gouvernement pour le renforcement de l’exercice du culte, se focalise sur
la construction et la mise à niveau de complexes religieux et culturels, la conservation des
mosquées historiques, ainsi que la restauration et la modernisation des infrastructures sociales
et culturelles. Les principaux projets y afférents, ayant été achevés au cours de l’année 2024,
se présentent dans le tableau ci-après :
                                                                                                   Coût
           Région                                         Projet
                                                                                                  (MDH)
                               Complexe religieux et culturel à Al Hoceima                         52

                               Restauration du Darih Moulay Touhami et ses annexes                 4,94

 Tanger-Tétouan-AI Hoceima     Restauration du Darih Boughaleb et ses annexes                       6

                               Restauration du Zaouia Tijania                                      4,1

                               Réhabilitation et valorisation de la Médina de Tanger                41

 Fès-Meknès                    Mise en valeur de la Médina de Fès                                  36

                               Institut royal pour la recherche sur l'histoire du Maroc            52

 Rabat-Salé-Kénitra            Complexe religieux, culturel et administratif à Salé                67,5

                               Restauration du Darih Sidi Ben Acher                                5,2
                                               NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


                                                                                                    Coût
             Région                                            Projet
                                                                                                   (MDH)
                                    Complexe religieux et culturel à Benguerir                      58,9
Marrakech-Safi
                                    Restauration du Darih Sidi Megdoul                              5,38

Drâa-Tafilalet                      Complexe religieux et culturel à Tinghir                        63,5

                                               Total                                               396,52

Par ailleurs, les principaux projets de construction des complexes religieux et culturels, de
restauration du patrimoine social et culturel et de construction de bâtiments administratifs,
programmés pour l’année 2025, se présentent comme suit :

                                                                                          Etat d’avancement
                                                                                            à fin juin 2024
                                                                 Coût      CP 2025
    Région                           Projet                                                            Taux
                                                                (MDH)       (MDH)
                                                                                       Emissions
                                                                                                   d’avancement
                                                                                        (MDH)
                                                                                                   physique (%)

                      Complexe religieux et culturel de
                                                                  64           1,93      55,7           90
Tanger-               Larache
Tétouan-AI
Hoceima
                      Conseil local des Oulémas à Tétouan         9,45         2,50       1,51          30

                  Complexe religieux et culturel de
L'Oriental                                                        57             13       4,3           25
                  Jerada

                  Amélioration du cadre de vie de                                                          —
                                                                  84,3           22      62,3
                  l’ancienne Médina de Fès

                      Complexe religieux et culturel du                                              Phase de
Fès-Meknès                                                        40             3       0,24
                      Hajeb                                                                         lancement

                      Complexe religieux et culturel de
                                                                  30             0,2       —       Phase d’étude
                      Missour

                      Institut Mohammed VI des lectures et
                                                                 64,06           14      14,95          40
                      études coraniques

Rabat-Salé-           Complexe religieux et culturel de Sidi
                                                                 68,20           4        0,6           40
Kénitra               Slimane

                      Complexe religieux et culturel de
                                                                  58           3,49     40,53           80
                      Kénitra

Béni Mellal-          Complexe religieux et culturel de
                                                                 73,20           25      17,86          80
Khénifra              Fquih Ben Saleh

Casablanca -          Complexe religieux et culturel de Sidi
                                                                  60,1         2,17       53            95
Settat                Bennour

                      Complexe religieux et culturel d’EI
                                                                  93,5         0,56       78            98
                      Kelâa des Sraghna
Marrakech-Safi
                      Complexe religieux et culturel
                                                                 73,00         2,41      26,84             75
                      d’Essaouira
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025


                                                                                          Etat d’avancement
                                                                                            à fin juin 2024
                                                            Coût         CP 2025
     Région                           Projet                                                             Taux
                                                           (MDH)          (MDH)
                                                                                       Emissions
                                                                                                     d’avancement
                                                                                        (MDH)
                                                                                                     physique (%)

                       Complexe religieux et culturel                                                   Phase de
 Drâa-Tafilalet                                             75,00               25        0,6
                       d’Errachidia                                                                    lancement

                       Complexe religieux et culturel
                                                              98               15,97     5,97               20
                       d’Agadir
 Sous-Massa
                       Complexe religieux et culturel de
                                                              75               1,20      0,06        Phase d’étude
                       Tiznit

                              Total                        1.022,81         136,43     362,46


En 2025, il est prévu la construction, la reconstruction et la mise à niveau d’un ensemble de
mosquées, dont les crédits sont répartis par région comme suit :

                                         Construction         Reconstruction                    Mise à niveau
              Région                     des mosquées          des mosquées                     des mosquées
                                               (MDH)                  (MDH)                        (MDH)

 Tanger-Tétouan-AI Hoceima                      5,43                   27,11                        31,55

 L’Oriental                                      —                    23,96                         31,37

 Fès-Meknès                                    29,90                   17,31                        23,10

 Rabat-Salé-Kénitra                            8,00                    8,09                         14,49

 Béni Mellal-Khénifra                           2,75                  28,60                         39,40

 Casablanca-Settat                             14,75                   9,04                         11,32

 Marrakech-Safi                                30,40                  14,54                         24,89

 Drâa-Tafilalet                                15,55                  35,23                         50,02

 Sous-Massa                                    28,14                  20,85                         29,61

 Guelmim-Oued Noun                              1,62                    —                            1,20

 Laâyoune-Sakia El Hamra                        6,17                   1,40                          1,59

               Total                           142,71                 186,13                       258,55

11.3.1.8 Programme de rattrapage des déficits en infrastructures et services de base
         dans les territoires sous-équipés, dans le cadre de l’INDH

Au titre de la mise en oeuvre du plan d’action de ce programme pour l’année 2024, 1.701
projets ont été validés, jusqu’à fin mai, pour un montant global d’investissement de
614,12 MDH. La ventilation par région de cette enveloppe se présente comme suit :
                                            NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                                         Nombre de          Crédits INDH               Bénéficiaires
              Région
                                       projets/actions       (en MDH)                   potentiels
Tanger-Tétouan-AI Hoceima                     102               81,81                       51.192
L'Oriental                                    251              42,68                       85.488
Fès-Meknès                                    150              98,49                       90.650
Rabat-Salé-Kénitra                            48               54,96                       157.994
Béni Mellal-Khénifra                          143              55,76                       38.736
Casablanca-Settat                             478              68,56                       80.694
Marrakech-Safi                                216              96,29                       205.922
Drâa-Tafilalet                                185              73,44                       109.197
Souss-Massa                                   94               27,99                       88.573
Guelmim-Oued Noun                             13                10,11                       1.570
Laâyoune-Sakia El Hamra                       21                4,03                        4.928

                 Total                       1.701             614,12                      914.944

De même, la ventilation des 1.701 projets par secteur peut être illustrée par le graphique
suivant :




                                                                ■ Désenclavement routier

                                                                ■ AEP

                                                                ■ Electrification




                  AEP : Alimentation en Eau Potable

Au titre de l’année 2025, et en perspective du lancement de la 4eme phase de l’INDH, le
programme de rattrapage des déficits en infrastructures et services sociaux de base, dans les
territoires sous équipés, sera doté d’un budget prévisionnel de 51 MDH, au profit de
71 préfectures et provinces, ciblant prioritairement 3 secteurs d’intervention ; à savoir : la
réalisation des pistes et des routes rurales, la desserte en eau potable et le raccordement en
électricité principalement par des Kits photovoltaïques.
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025


11.3.1.9 Habitat et politique de la ville et aménagement du territoire
    ► Habitat
En vue de lutter contre la prolifération de l’habitat insalubre qui compromet le tissu urbain des
villes marocaines, le Gouvernement intensifie ses efforts pour atteindre les objectifs du
développement fixés à l’horizon 2030. Dans ce contexte, les régions connaîtront un rythme
d’investissement plus accéléré pour permettre l’éradication totale des bidonvilles à l’horizon
2028, et entamer la mise en oeuvre des actions d’amélioration du cadre de vie et des
conditions d’habitabilité des ménages.
La nouvelle approche impliquant le secteur privé dans le domaine de la lutte contre l’habitat
insalubre, a créé un tournant dans la méthode empruntée ces dernières années. Elle a permis
d’accélérer le rythme de résorption et de maîtrise des délais, et de renforcer davantage la
capacité d’acquisition de logements par les bénéficiaires.
Au titre de l’année 2025, le Gouvernement poursuivra le lancement des projets visant
l’amélioration des conditions de l’habitat de la population, à travers tout le territoire du
Royaume, et ce en adoptant une approche conventionnelle impliquant l’ensemble des acteurs,
qui vise à assurer la mise en oeuvre effective des programmes et l'accompagnement social de
la population bénéficiaire. À ce titre, le nombre de conventions conclues dans ce cadre et
dont les projets y afférents sont en cours de réalisation, totalisent 521 conventions bénéficiant
à environ 1.435.707 ménages, pour un investissement global de plus de 53 MMDH.
La répartition régionale de ces projets d’investissement, par programme, est détaillée dans les
tableaux ci-après :

   Programme 1 : Ville Sans                                                              Reliquat
     Bidonvilles « VSB »                                         Nombre de     Coût      2024 &      CP 2025
                                                 Région
                                                                  ménages     (MDH)      ultérieur    (MDH)
  Objectifs: Éradication de tous                                                          (MDH)
  les bidonvilles des centres            Tanger-Tétouan-AI
  urbains.                                                         1.699      101,36      20,18       20,18
                                         Hoceima
  Consistance:         T rois modes
  opératoires ont été retenus            L’Oriental               14.496     1.205,38     85,06       70,00
  pour       la    résorption      des
  bidonvilles, il s’agit de :            Fès-Meknès                7.339     1.056,83     57,06       56,76
• La restructuration : cette
  opération a pour objectif de           Rabat-Salé-Kénitra       100.653    16.186,66   874,47       681,55
  doter les grands et moyens
  bidonvilles en équipements
  d’infrastructure nécessaires, et       Béni Mellal- Khénifra     2.586      163,94       9,70       9,00
  de régulariser leur situation
  urbanistique et foncière.
                                         Casablanca-Settat        76.449     8.909,97     315,25      297,61
• Le relogement : ce mode
  d’intervention      consiste      en   Marrakech-Safi           25.448     1.328,30     36,80       33,39
  l’attribution     de      logements
  sociaux                    destinés
                                         Souss-Massa               4.159      617,68      14,95       10,00
  essentiellement aux ménages
  des bidonvilles des grandes et         Laâyoune-Sakia EL
  moyennes agglomérations.                                         2.800      532,00      26,88       21,51
                                         Hamra
• Le recasement : il consiste en
  l’attribution de lots aménagés à       Multirégional              —           —           —           12
  valoriser en auto-construction
  assistée.                              Total                    235.629    30.102,12   1.440,35     1.212
                                                NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT




                                                                                        Reliquat
     Programme 2 : Habitat                                    Nombre de      Coût       2024 &      CP 2025
                                              Région
                                                               ménages      (MDH)       ultérieur    (MDH)
        Menaçant Ruine                                                                   (MDH)
           « HMR »                     Tanger-Tétouan-AI
                                                                9.833       588,57       99,45       10,00
                                       Hoceima
                                       L’Oriental               4.919       361,95       57,33       7,00
 Objectifs: Lutte contre le
 délabrement des constructions         Fès-Meknès              75.798      1.480,68      176,29      42,00
 et des habitations menaçant
 ruine, afin de préserver la vie       Rabat-Salé-Kénitra       4.984       612,21       67,45       10,00
 des habitants et citoyens, ainsi
 que la préservation et la             Béni Mellal-Khénifra     7.706       135,12        7,73        —
 valorisation    du     patrimoine
 architectural dans les villes.        Casablanca-Settat        8.907      1.800,07      122,68      15,00

 Consistance: Relogement,              Marrakech-Safi           13.226      875,05       191,05      11,00
 recasement et restructuration
                                       Drâa-Tafilalet           3.663       94,27         4,02        —
 des ménages concernés par la
 démolition de leurs habitations.      Souss-Massa             10.608      300,60        53,75       5,00

                                               Total           139.644     6.248,52     779,75        100


    Programme 3 : Mise à                                                                Reliquat
                                                              Nombre de       Coût      2024 &      CP 2025
      Niveau Urbaine et                       Région
                                                               ménages       (MDH)      ultérieur    (MDH)
     Restructuration des                                                                 (MDH)
   Quartiers d’Habitat Non             Tanger-Tétouan-AI
                                                                94.707      1.278,00     109,93      80,00
   Réglementaires ou Sous              Hoceima
           Équipés                     L’Oriental               140.399     3.395,85     403,00      10,00
                                       Fès-Meknès               200.912     1.526,35     124,49       7,00
  Objectifs:
• Lutte contre l’exclusion en milieu   Rabat-Salé-Kénitra       93.637      1.168,27      112,35      2,00
  urbain ;                             Béni Mellal-Khénifra     141.823     1.286,20     43,387       3,00
• Amélioration du cadre bâti et de     Casablanca-Settat        95.352      2.209,79     934,96      40,00
  la qualité des espaces urbains
  dans les villes.                     Marrakech-Safi           116.609     2.547,09     410,04      43,06
                                       Drâa-Tafilalet           50.657      459,06        28,28       6,00
  Consistance :
• Généralisation de l’accès aux        Souss-Massa              86.188       959,76       51,16       1,00
  équipements et aux infrastructures
  de base au profit des quartiers      Guelmim-Oued Noun         7.560      145,825       12,39       6,94
  sous équipés ;                       Laâyoune-Sakia EL
                                                                 6.580      463,30        117,34     21,00
                                       Hamra
• Travaux des routes internes des
  quartiers non réglementaires, et     Multirégional               —           —            —          29
  aménagement des espaces verts                Total           1.034.424   15.439,495   2.347,327     249
  et des espaces publics.
 En ce qui concerne les projections d’investissement pour la période 2025-2027, l’effort
 budgétaire sera principalement orienté vers le lancement de nouveaux projets visant la
 résorption des bidonvilles au niveau des régions de Casablanca-Settat, Marrakech-Safi,
 Tanger-Tétouan-AI Hoceima, Fès-Meknès, et Laâyoune-Sakia El Hamra, ainsi que vers le
 traitement de l’habitat menaçant ruine et des tissus anciens.
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


     ► Politique de la ville
En parallèle avec les efforts déployés pour la lutte contre la prolifération de l’habitat insalubre,
les actions du programme de la politique de la ville connaîtront une accélération du rythme
d’investissement, afin d’atteindre les objectifs assignés à l’horizon 2030. En effet,
l’organisation prévue de la coupe du monde sera un catalyseur pour mettre à niveau nos
territoires, et contribuer au rehaussement de leur attractivité pour leur permettre de jouer un
rôle important dans le déclenchement d’une dynamique économique, et par conséquent la
création de l’emploi et du bien-être des citoyens.

                               Programme de la politique de la ville


Objectifs : Amélioration de l’accès des populations aux infrastructures de base et aux
équipements publics et services de proximité et renforcement de l’intégration urbaine.
Consistance du programme :

 - Le désenclavement des quartiers par leur desserte en voirie et en réseaux divers (eau,
   électricité, assainissement des eaux usées et des eaux pluviales) et l’amélioration de la
   connectivité des unités urbaines des villes par l’aménagement des routes inter-quartiers;
 - La réalisation des travaux d’éclairage public ;
 - La création des espaces récréatifs, des places publiques, des placettes, des espaces verts,
   des aires de jeux et des équipements de proximité (terrains et installations sportifs,
   équipements culturels, salles polyvalentes, points de lecture, maisons des jeunes, etc.) ;
 - L’aménagement et l’extension des espaces piétons (rénovation, élargissement et
   aménagement paysager des trottoirs des avenues urbaines, aménagement des coulées
   vertes, aménagement des berges et des cours d’eau intra-urbain, mobilier urbain).
                                                Coût           Reliquat 2024 &       CP 2025
                  Région
                                               (MDH)           ultérieur (MDH)        (MDH)
 Tanger-Tétouan-AI Hoceima                     7.618,26            414,52              48,36

 L’Oriental                                   3.869,55             753,661             95,52

 Fès-Meknès                                   5.046,53             1.126,48            38,97

 Rabat-Salé-Kénitra                           5.572,75             1.113,12            122,14

 Béni Mellal-Khénifra                          879,00              453,87              58,03

 Casablanca-Settat                            1.542,20             342,25              32,45

 Marrakech-Safi                               3.696,42             1.338,31            64,72

 Drâa-Tafilalet                                1.601,11            187,54              24,15

 Souss-Massa                                  10.436,34            923,89              55,97

 Guelmim-Oued Noun                              511,49             212,50              37,34

 Laâyoune-Sakia El Hamra                       155,06              23,055              4,79

 Dakhla-Oued Ed-Dahab                         1.230,47             243,55              38,58

 Multirégional                                    —                    —                105

                  Total                       42.159,18          7.132,746            726,02
                                     NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


11.3.2 L’investissement dans les infrastructures en riposte aux répercussions néfastes
      des changements climatiques
L'investissement dans les infrastructures joue un rôle primordial dans le développement
économique, social et environnemental du pays. Il contribue au renforcement de son
attractivité économique, et soutient les dynamiques locales en facilitant l'inclusion sociale et
en réduisant les disparités régionales.
En matière d’infrastructures hydrauliques, et afin de répondre aux enjeux liés à la sécurité
alimentaire et énergétique, le Gouvernement, en application des Hautes Directives de Sa
Majesté le Roi, que Dieu L'assiste, à l'occasion du vingt-cinquième anniversaire de Son
accession au trône, s’engage à garantir l'accès à l'eau potable pour l'ensemble de la
population, et à satisfaire 80 % des besoins en irrigation au niveau national.
Il est à souligner que Sa Majesté a insisté dans son discours sur la nécessité d’une mise à jour
continue des instruments de la politique de l’eau au Maroc, tout en appelant à l’accélération de
la réalisation des projets de construction des barrages, des grands projets de transfert d’eau
entre les bassins hydrauliques et des stations de dessalement de l’eau de mer. Le Souverain a
également appelé à développer une industrie nationale de dessalement de l’eau de mer, à
créer des filières de formation spécialisées pour les ingénieurs et les techniciens, notamment
dans les techniques de dessalement de l’eau de mer, et à encourager la création d’entreprises
nationales spécialisées dans la réalisation et l’entretien des stations de dessalement.
11.3.2.1 Secteur de l’Eau
Aujourd’hui, le Maroc dispose d’un réseau de 154 grands barrages d’une capacité totale de
plus de 20,7 milliards de m3, et de 146 petits barrages, permettant ainsi de répondre aux
besoins en eau (potable, industrielle et touristique) dans des conditions optimales, d’assurer
une irrigation à grande échelle (sur plus de 2 millions d’hectares), et de contribuer à la
production d’énergie hydroélectrique (à hauteur de 4% à 10% en termes de besoins en
électricité) et à la protection contre les phénomènes extrêmes tels que les inondations et les
sécheresses.
En 2024, deux grands barrages ont été mis en eau, il s’agit du barrage Mdez dans la province
de Sefrou et du barrage Fask dans la province de Guelmim. Aussi, l’année 2024 connaîtra
l’achèvement du barrage Koudiat Borna dans la province de Sidi Kacem et la mise en eau du
barrage Ghiss dans la province d’AI Hoceima, ainsi que la poursuite des travaux de 15 autres
grands barrages, à savoir :
  - Barrages Sidi Abbou (Taounate) et Béni Azimane (Driouch) dont l'achèvement est prévu
    en 2025 ;
  - Barrages Ait Ziat (Al Haouz), Targa Oumadi (Guercif), Boulaouane (Chichaoua),
    surélévation du barrage Mokhtar Soussi (Taroudant), Tamri (Agadir), Kheng Grou
    (Figuig), Sakia El Hamra (Laâyoune), et la surélévation du barrage Mohamed V
    (l’Oriental) dont l’achèvement est prévu en 2026 ;
  - Barrages Oued Lakhdar (Azilal) et Taghzirt (Béni Mellal), dont l'achèvement est prévu en
    2027 ;
     Barrage Ratba (Taounate) dont l’achèvement est prévu en 2028 ;
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


    - Barrage Ribat El Kheir (Sefrou) et la surélévation du barrage Imfout (Settat) dont
      l’achèvement est prévu en 2029.
Les travaux d’interconnexion entre la retenue du barrage Garde du Sebou et celle du barrage
Sidi Mohammed Ben Abdellah sur le Bouregreg, intégrés à la première phase du projet
d’interconnexion Sebou-Bouregreg-Oum Er-Rbia dans le cadre du Programme National
d’Approvisionnement en Eau Potable et d’irrigation (PNAEPI) 2020-2027, ont été achevés et
progressivement mis en service à partir d’août 2023. En outre, les travaux d’interconnexion
entre la retenue du barrage Oued El Makhazine et celle du barrage Dar Khrofa, pour la
sécurisation de l’alimentation en eau potable du Grand Tanger, sont en cours de réalisation.
Par ailleurs, la première tranche du projet de dessalement de l’eau de mer de la ville d'Agadir a
été finalisée et mise en service, et les efforts se poursuivront pour achever la seconde phase
avant la fin de l’année 2026.
Les tableaux ci-dessous présentent la répartition régionale des projets d'investissement les
plus importants dans le secteur de l'eau, programmés au titre de l'année 2025 :

  ► Construction des grands barrages (projets en cours de réalisation):
                                                                                       Situation à fin Juin
                                                                    Coût                      2024
                                                                            CP 2025
           Région                        Grand barrage             global
                                                                             (MDH)    Emissions
                                                                                                    Taux de
                                                                  (MDH)*                           Réalisation
                                                                                       (MDH)
                                                                                                     (en %)
 Tanger-Tétouan-AI
                                Barrage sur Oued Ghiss             902        20       888,3         98,50
 Hoceima
                                Barrage Targa Oumadi               920        250       516,8          51
                                Surélévation du Barrage
                                                                   885       280        206            39
 L'Oriental                     Mohamed V
                                Barrage Béni Azimane               1.167      316        611         54,50

                                Barrage Kheng Grou                 940       280        339            38

                                Barrage M'dez                      414        64        250            96
                                                                                                    T ravaux
                                Barrage Ribat Al Kheir             1.077      130        20
 Fès-Meknès                                                                                         en cours
                                Barrage Sidi Abbou                 657        82        368            70

                                Barrage Ratba                     2.990      484        953            22

 Rabat-Salé-Kénitra             Barrage Koudiat Borna               815       53        555            84

                                Barrage Taghzirt                   1.053      150       25,3          7,50
 Béni Mellal-Khénifra
                                Barrage sur Oued Lakhdar           1.240     300        173            27

                                Barrage Ait Ziat                   1.029      218       485            60
 Marrakech-Safi
                                Barrage Boulaouane                 925        168       398            49

                                Barrage Tamri                      1.899      542       656            44
 Souss-Massa                    Surélévation du Barrage Mokhtar
                                                                   1.192      265       552          48,50
                                Soussi
 Laâyoune-Sakia El              Barrage Sakia El Hamra -
                                                                   304        130       240            74
 Hamra                          Achèvement
                                Total                             18.409     3.732    7.236,4           -
(*) Coût global (génie civil)
                                            NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


  ► Construction des moyens barrages (projets en cours de réalisation):

                                       Nombre de barrages             Coût global           CP 2025
                 Région
                                            moyens                      (MDH)*               (MDH)

 Casablanca-Settat                                1                        265                40

 Marrakech-Safi                                   1                        580                150

 Souss-Massa                                      2                        650                95

                  Total                           4                        1.495              285

(*) Coût global (génie civil)

     ► Construction de nouveaux barrages :

                                                                      Coût global           CP 2025
                 Région                Nombre de barrages
                                                                        (MDH)                (MDH)

 Tanger-Tétouan-AI Hoceima                        2                        3.500              80

 L’Oriental                                       1                        600                20

                  Total                           3                        4.100              100

En termes d’investissement dans les technologies de dessalement de l’eau de mer et dans
l’interconnexion des bassins hydrauliques, il est prévu, en 2025, le lancement et la poursuite
de la mise en œuvre des projets stratégiques ci-après :
                                                                                     Coût      CP 2025
              Région                                  Projet
                                                                                    (MDH)       (MDH)

                                Ligne électrique pour la station de
 Casablanca-Settat                                                                   259        135,20
                                dessalement du Grand Casablanca

                                Projet de dessalement de l'eau de mer de
 Laâyoune-Sakia El Hamra                                                             480            230
                                Laâyoune

                                Projet de dessalement de l'eau de mer de
 Dakhla-Oued Ed-Dahab                                                                450            245
                                Dakhla

                                    Total                                           1.189          610,2

11.3.2.2 Réseau routier
Au titre de l’année 2025, et dans le cadre de la mise en œuvre de la stratégie
gouvernementale pour le développement et la conservation des infrastructures routières, le
projet de loi de finances prévoit le maintien des efforts d’investissement pour la réalisation des
projets routiers visant la conservation du patrimoine routier, la réhabilitation des ouvrages
d’art, la modernisation du réseau routier structurant, ainsi que la signalisation et l’amélioration
de la sécurité routière.
Les principaux investissements programmés en 2025 s’articulent autour de 5 grands projets
stratégiques et 3 programmes routiers, à savoir :
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025


                                                                                                             Etat d’avancement à
                                                                                                  CP            fin juillet 2024
                                                                       Ligne         Coût
                     Projet                            Région                                    2025                       Taux de
                                                                       (Km)         (MDH)                  Emissions
                                                                                                (MDH)                      réalisation
                                                                                                            (MDH)
                                                                                                                             (en %)
 Projet de la voie de contournement
                                                   Souss- Massa          27          1.402        50          809              90
 Nord-Est du grand Agadir
 Autoroute Guercif-Nador pour
 accompagner le projet du port Nador-                                   104         7.400        752          560              13
 West Med                                            L’Oriental
 Connectivité routière Nador-West Med                                  225,3         2.192       100           398            69*

 Dédoublement de la RN8 entre Fès et
 Taounate sur 73 km avec reconstruction
 et dédoublement de 11 ouvrages d’art               Fès-Meknès           73          1.560       250           133            45*
 (1er et 2ème lot sur 35 km en cours de
 réalisation)


 Projet de mise à niveau de la RN6 entre
                                                    Rabat-Salé -
 l’aéroport de Rabat et Sidi Allai El                                    20           991         121          538             90
                                                      Kénitra
 Bahraoui


                               Total                                      -         13.545      1.273        2.438                 -
 (*) C'est le taux de réalisation des sections dont les travaux sont en cours.




           Programme de maintenance routière



                                                                                                                          CP* 2025
  Objectifs : Conservation du patrimoine routier et                                          Région                        (MDH)
  amélioration de la qualité de service du réseau routier.
                                                                               Tanger-Tétouan-AI Hoceima                     58
  Consistance : Travaux d’entretien des routes revêtues.
                                                                               L’Oriental                                    17
  Etat d’avancement : Une légère amélioration de l'état
  du réseau routier d’environ un point par rapport à 2020,                     Fès-Meknès                                    78
  soit un taux de 63,9 % des routes en état moyen à bon.                       Rabat-Salé -Kénitra                           54
  Au titre de l’année 2025, il est prévu :
                                                                               Béni Mellal-Khénifra                          81
  ■   La consolidation des opérations en cours
      d’exécution à travers la mobilisation de 329 MDH en                      Marrakech-Safi                                113
      termes de crédits de paiement au niveau du Budget                        Casablanca-Settat                             78
      Général (BG), et 835 MDH au niveau du Fonds
      Spécial Routier (FSR).                                                   Drâa-Tafilalet                                23
  ■   La réalisation de nouveaux projets de maintenance                        Souss-Massa                                   65
      routière pour 1.062 Km de routes avec un coût
      global de 1,93 MMDH. Ce programme sera financé                           Guelmim-Oued Noun                             28
      en totalité par le FSR, et le montant des crédits de                     Laâyoune-Sakia El Hamra                       30
      paiement programmé en 2025 est de 643 MDH.
                                                                               Dakhla-Oued Ed-Dahab                          18
  La répartition régionale du plan d’action prévisionnel de
  l’année 2025 pour le programme de la maintenance                                           Total                          643
  routière se présente ci-contre :                                            (*) Imputés sur le Fonds Spécial Routier.
                                            NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT




           Programme de réhabilitation
               des ouvrages d’art


 Objectifs : Préserver le parc national d’ouvrages d’art
 (OA) et assurer la fluidité du trafic sur les routes.
                                                                                                         CP* 2025
 Consistance : Reconstruction/construction de nouveaux                       Région
                                                                                                          (MDH)
 ouvrages.
 Etat d’avancement : Le Royaume dispose d'un parc               Tanger-Tétouan-AI
                                                                                                           25
                                                                Hoceima
 important et diversifié de ponts et d’ouvrages d’art
 d’environ 15.713 unités. Aujourd’hui, ces ouvrages             L’Oriental                                 45
 souffrent de l'obsolescence naturelle de leurs matériaux       Fès-Meknès                                 90
 constitutifs et de leur faible capacité portante, d’où
 l’effort d’investissement qui leur sera accordé en 2025        Rabat-Salé-Kénitra                          51
 pour faire face à cette difficulté.
                                                                Béni Mellal-Khénifra                        16
 Au titre de l’année 2025, il est prévu :
   ■   La consolidation des opérations en cours                 Marrakech-Safi                             36
       d’exécution à travers la mobilisation de 46 MDH en       Casablanca-Settat                           15
       termes de crédits de paiement au niveau du
       Budget Général (BG), et de 152 MDH au niveau du          Drâa-Tafilalet                             49
       Fonds Spécial Routier (FSR).
                                                                Souss-Massa                                 17
   ■   La réalisation de nouveaux projets routiers visant la
       construction de 68 OA avec un coût global de             Guelmim-Oued Noun                           19
       491 MDH. Ce programme sera financé en totalité
                                                                             Total                         363
       par le FSR, et le montant des crédits de paiement
       programmé en 2025 est de 165 MDH.                       (*) Y compris te Fonds Spécial Routier.


 La répartition régionale du plan d’action prévisionnel de
 l’année 2025 pour le programme de réhabilitation des
 ouvrages d’art se présente ci-contre :


           Programme de construction des voies-express dans les plans de développement
                                        régionaux (PDRs)


Objectifs :Améliorer la connectivité routière aux principales infrastructures et centres
économiques du pays.
Consistance : Construction de nouvelles voies-express et extension et maintenance des voies
existantes.
                                                                                           CP 2025
                                                                                            (MDH)
Fes - Meknes                                                                                  290
Marrakech-Safi                                                                                 50
PAhAt-ÇAlp-kpnitrA                                                                             1R1
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


11.3.2.3 Réseau portuaire
Au titre de l’année 2025, le Gouvernement envisage la poursuite de la réalisation des projets
d’investissement dans le secteur portuaire et maritime, et ce à travers la mise en œuvre de son
programme stratégique qui traduit la volonté affirmée de positionner le Maroc, en tant
qu'acteur clé du commerce maritime au niveau local et international, tout en assurant la
durabilité et la sécurité du littoral.
Au titre de l’année 2025, la mise en œuvre des principaux projets et opérations se poursuivra,
selon la répartition régionale ci-après :
                                                                                        Etat d’avancement à fin
                                                                   Coût      CP 2025       septembre 2024
       Région                          Projet                                                         Taux de
                                                                 (en MDH)     (MDH)     Emissions
                                                                                                     réalisation
                                                                                         (MDH)
                                                                                                       (en %)

                      Projet d’extension du port de pêche
                                                                    348       91,09       145,38         68
                      Jebha (2ème phase)
Tanger-Tétouan-AI
Hoceima
                      Projet de protection du littoral de la                   20,7       28,82          87
                                                                   77,3
                      ville de Larache


Rabat-Salé-Kénitra    Projet de protection du littoral de Salé     563,8        60          —       Phase d’étude


                      Travaux de prolongement de la digue
Casablanca-Settat                                                 1.183,7      158       540,82          60
                      Moulay Youssef au port de Casablanca

                      Projet de confortement de la falaise de
Marrakech-Safi                                                      135,7      33,7        6,66           5
                      Jorf Amouni à Safi

                      Projet du nouveau port de Dakhla
                                                                  13.614,4    2.161,3    2.394,03        25
                      Atlantique
Dakhla-Oued Ed-
Dahab
                      Projet de l’épi d’arrêt de sable au port
                                                                   125,5       10,9       83,84          83
                      de Lamhiriz

                           Total                                 16.048,4    2.535,69    3.199,55         -


De même, plusieurs nouveaux projets et opérations seront lancés en 2025, notamment ceux
visant la préservation de l'environnement et la sécurité de la navigation maritime, avec une
enveloppe budgétaire de 35 MDH, ainsi que ceux concernant la délimitation, la préservation et
la valorisation du domaine public maritime et portuaire, avec un montant de 16 MDH imputé
au niveau du Compte d’Affectation Spécial concerné.
11.3.2.4 Réseau du transport et de la logistique
Durant la période 2024-2026, et en partenariat avec toutes les parties prenantes impliquées,
le Gouvernement s’emploie activement à orienter les investissements publics dans le secteur
des transports et de la logistique, vers les opérations prioritaires suivantes :
   ■   Le développement et la diversification des infrastructures de transport durables ;
   ■   La réduction des coûts de production liés à la logistique et l’amélioration de la qualité
       des services ;
                                             NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


    ■    Le développement de zones logistiques à prix compétitifs ;
    ■    La numérisation des activités du secteur et l’introduction de technologies modernes ;
    ■    Le développement des métiers du secteur et la qualification des employés ;
    ■    Le développement des compétences et la valorisation de la formation.
Ces investissements seront déployés selon une approche visant à renforcer les contributions
des établissements publics, ainsi que le partenariat avec le secteur privé et les collectivités
territoriales, dans le cadre d’une démarche de contractualisation avec l'État pour la
mobilisation du financement aux grands projets stratégiques.
Dans ce sens, un cadre contractuel a été établi entre l'État et l'Office National des Chemins de
Fer (ONCF), pour le financement et la réalisation des projets d’investissement d’envergure
dans le secteur ferroviaire, notamment le projet de la liaison ferroviaire du port Nador-West
Med et le projet d’extension de la ligne à grande vitesse Kénitra-Marrakech et Marrakech-
Agadir, dont les crédits programmés au titre de l’année 2025 se présentent comme suit :
                                                                                                  Etat d’avancement à
                                                                                                      fin juin 2024
                                                                      Coût         CP 2025
        Région                        Projet                                                                    Taux de
                                                                     (MDH)          (MDH)         Emissions
                                                                                                               réalisation
                                                                                                   (MDH)
                                                                                                                 (en %)

                   Poursuite de la réalisation des études et
 L’Oriental        des opérations d'expropriation du projet de         733          203             180            17
                   liaison ferroviaire du port Nador-West Med

 Marrakech-Safi    Etudes pour les projets d'extension de la
                   ligne à grande vitesse Kénitra-Marrakech et        1.442         400             377           48,4
 Souss-Massa       Marrakech-Agadir

                            Total                                     2.175         603             557             -

Par ailleurs, le Gouvernement poursuit la mise en œuvre, sur la période 2021-2024, d'un
ensemble de projets d’investissement en infrastructure aéroportuaire dont : la construction
d'une nouvelle aérogare à l'aéroport de Rabat-Salé, l'aménagement et la réhabilitation de la
zone centrale de l'aéroport Mohammed V à Casablanca, l'extension de l'aéroport Sania R'mel
de Tétouan avec l’augmentation de sa capacité, ainsi que l'élargissement de l'aéroport Chérif
El Idrissi dans la province d'AI Hoceima, et l'extension de l'aire aéronautique et la construction
d'une nouvelle tour de contrôle et d'un nouveau terminal passagers à l'aéroport de Nador
El Aroui.
Les principaux projets d’infrastructure aéroportuaire prévus au titre de l’année 2025, se
présentent comme suit :

                                                                                        Coût                  CP 2025
        Région                                 Projet
                                                                                       (MDH)                   (MDH)

 Tanger-Tétouan-   Développement des           installations   terminales     de
                                                                                          1.541                661*
 Al Hoceima        l’aéroport de Tanger

                   Développement des           installations   terminales     de
 Fès-Meknès                                                                               500                  480*
                   l’aéroport de Fès Saïss
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025


                                                                                                    Coût           CP 2025
       Région                                         Projet
                                                                                                   (MDH)            (MDH)

                        Développement des installations                 terminales     de
 Marrakech-Safi                                                                                    1.846              861*
                        l’aéroport de Marrakech

                        Développement des             installations     terminales     de
 Souss-Massa                                                                                       1.530             600*
                        l’aéroport d’Agadir

                                          Total                                                    5.417             2.602
(*) Seul le volet « expropriation » de ces projets est imputé sur les crédits du Budget Générai.

Dans le cadre de la mise en œuvre de la stratégie nationale logistique au niveau régional, le
Gouvernement œuvre à mobiliser les fonds nécessaires pour la réalisation des projets de
développement des installations logistiques, en cohérence avec les plans de développement
régionaux élaborés en collaboration avec les acteurs locaux. À ce propos, il convient de
souligner que la mise en place de tels projets a débuté en 2021 avec le lancement du premier
projet de zone logistique du sud d'Agadir, dans la région de Souss-Massa.
Au titre de l’année 2025, le Gouvernement prévoit la poursuite des projets en cours de
réalisation et le lancement de nouvelles infrastructures logistiques dans plusieurs régions du
Royaume, dont les zones logistiques à Ait Melloul (45 hectares), à Ras El Ma (32 hectares), à
Ouled Saleh (70 hectares) et à Béni Mellal (9 Hectares).
La répartition régionale des principaux projets prévus au titre de l’année 2025, se présente
comme suit :
                                                                                                    Situation à fin juin 2024
                                                                        Coût         CP 2025
      Région                             Projet                                                                   Taux de
                                                                       (MDH)          (MDH)        Emissions
                                                                                                                 réalisation
                                                                                                    (MDH)
                                                                                                                   (en %)

                       Zone logistique de Ras El Ma (32
Fès- Meknès                                                              300            50            16,5             5
                       Hectares)

Rabat-Salé-            Zone logistique de Kénitra (45
                                                                         250            45           50,83            10
Kénitra                Hectares)

Béni Mellal-           Zone logistique de Béni Mellal (9                                               —        Lancement des
                                                                          50             10
Khénifra               Hectares)                                                                                   travaux

Casablanca-            Zone logistique d’Ouled Saleh (70                                                        Lancement des
                                                                         550            100            —
Settat                 Hectares)                                                                                   travaux

                       Zone logistique d’Ait Melloul (45
Souss-Massa                                                              350           45,9          120,19           84
                       Hectares)

                              Total                                     1.500         250,9          187,52            -
                                        NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


11.3.2.5 Programmes environnementaux
    ► Projets de Prévention et de Lutte contre la Pollution des Secteurs industriels et
      artisanaux
Ce programme vise la mise à niveau, sur le plan environnemental, des zones industrielles (ZI)
et artisanales à l’horizon 2035, dans le cadre d’une approche partenariale impliquant tous les
acteurs concernés. En 2025, la mise à niveau des Zones Industrielles de Skhirat et d'Aïn Attiq
se poursuivra, avec une enveloppe budgétaire d'environ 10 MDH.
    ► Projets de Protection et de Valorisation des Milieux Environnementaux
Le programme met en avant, selon une approche territoriale, les écosystèmes locaux, et
encourage une gestion durable des ressources naturelles. Ainsi, chaque région bénéficie dans
le cadre de ce programme, de projets spécifiques prenant en considération ses propres
atouts. Les projets programmés en 2025 se présentent comme suit :
                                                                                             CP 2025
        Région                                        Projet
                                                                                              (MDH)
                     Réhabilitation des parcs Ain Vital et Ajaabou à Ifrane                     5
 Fès-Meknès
                     Aménagement et valorisation du Parc National d'Ifrane                      5

 L’Oriental          Valorisation des déchets miniers et réhabilitation des sites à Jerada      1

 Casablanca-Settat   Aménagement paysager du lac El Oulfa à Casablanca                          5
                                          Total                                                16

Dans le même contexte, le Gouvernement élabore des conventions visant à améliorer la mise à
niveau environnementale du secteur oléicole, avec la contribution des partenaires concernés
au niveau régional, tout en ciblant les objectifs de la stratégie nationale de développement
durable. Les projets appuyés dans ce sens se présentent dans le tableau ci-après :
                                                                                             CP 2025
        Région                                        Projet
                                                                                              (MDH)
 Fès-Meknès          Traitement des margines dans le cadre du programme de                     7,6
                     développement régional (PDR)
Tanger-Tétouan-AI    Traitement des margines dans le cadre du programme de                     4,4
Hoceima              développement régional (PDR)
                                          Total                                                 12

    ► Programme National de Surveillance de la Qualité de l’air
Il consiste à mettre en place les réseaux régionaux pour la surveillance de la qualité de l’air,
conformément à la réglementation relative aux normes de la qualité de l'air et les modalités de
surveillance de l’air. Cela s’inscrit dans le cadre des efforts déployés pour répondre aux
directives de l’Organisation Mondiale de la Santé (OMS) en la matière.
En 2025, il est prévu le renforcement des réseaux existants, notamment à travers la
programmation des actions de partenariat avec les acteurs impliqués au niveau régional (La
Direction Générale des Collectivités Territoriales du Ministère de l’intérieur et les Conseils
Régionaux), dont la contribution du Budget de l’Etat (Département du Développement
Durable) atteint 9,4 MDH.
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


11.3.3 Le programme d’investissement dans les secteurs productifs
11.3.3.1 Agriculture
La stratégie « Génération Green 2020-2030 » pour le développement du secteur agricole,
s'inscrit dans une dynamique de transformation profonde, qui, en plus de l’amélioration in fine
des conditions de vie en milieu rural, vise à transformer les territoires en véritables moteurs de
développement économique.
Dans ce cadre, l’année 2024 a été marquée par la mise en œuvre d’un ensemble
d’investissements structurels, relevant de trois grands programmes. Ces projets seront
poursuivis durant l’année 2025, selon la répartition régionale ci-après :
    ► Programme de l'irrigation et de l'aménagement de l'espace agricole
        -     Projets en cours de réalisation :
                                                                                  Situation à fin juin 2024
                                                                 Coût   CP 2025                   Taux de
       Région                     Intitulé du projet                              Emissions
                                                                (MDH)    (MDH)                   réalisation
                                                                                   (MDH)
                                                                                                   (en %)
                        Projet d’Appui au Programme
 Tanger-Tétouan-AI      National d’Economie d’Eau en
                                                                 470      16,9     439,57            93
 Hoceima                Irrigation (2ème tranche)
                        (PAPNEEI2)
                        Projet de modernisation collective
 L’Oriental             du système d’irrigation de la plaine     870     342,21     291,72           42
                        du Garet
                        Projet d’aménagement hydro­
                        agricole (AHA) pour la sauvegarde       9.000    1.200      7.257            81
                        de la plaine de Saiss
 Fès-Meknès
                        Projet Moyen Sebou Inaouen Aval -
                                                                1.200     62         1.118           99
                        2ème tranche

                                                                                  Démarrage prévu après la
                        Projet d’aménagement hydro­                                signature de l'accord de
                        agricole (AHA) de la plaine du Gharb    7.000     700         prêt avec l’Agence
                        sur 30.000 ha                                             Japonaise de Coopération
                                                                                     Internationale (JICA)
                        Projet d’aménagement hydro­
 Rabat-Salé-Kénitra                                                                   Démarrage prévu
                        agricole (AHA) du périmètre Oualjet      342      100
                                                                                       incessamment
                        Essoltane

                        Modernisation des réseaux au niveau
                                                                 230     40,87     90,85            43
                        du Gharb

                        Projet de Résilience et Durabilité de
 Béni Mellal-Khénifra                                            750      254      113,78           15
                        l’irrigation « REDI »

                        Projet de reconversion à l'irrigation
                        localisée du secteur Amont T2 du
 Marrakech-Safi                                                  830      60        332             60
                        Périmètre de la Tessaout Aval,
                        Sultania Agafay et P2
                                                NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


                         Projet de Développement de
                         l’irrigation et d’Adaptation de
 Drâa-Tafilalet          l’Agriculture Irriguée aux                1.033       37        965            96
                         Changements Climatiques à l’aval du
                         Barrage Kaddoussa "PDIAAI-CC”

                         Projet de Résilience et Durabilité de      739       93,6
 Souss-Massa                                                                             17,66           15
                         l’irrigation « REDI »

                              Total                                22.464   2.906,58   10.625,58         -

        - Nouveaux projets en partenariat public-privé :
                                                                                        Coût          CP 2025
              Région                                      Projet
                                                                                       (MDH)           (MDH)
                                  Projet Sidi-Rahhal : Sauvegarde de l'agriculture
 Casablanca-Settat                                                                      625             100
                                  maraîchère
                                  Projet Oriental : Sauvegarde de 25 000 ha de la
 L’Oriental                       grande hydraulique de Moulouya et extension          3.500           300
                                  de 5 000 ha de nouvelles terres agricoles.
                                  Projet Souss-Massa : Combler le déficit hydrique
                                  dont souffre les périmètres de Tiznit et             13.000          300
 Souss-Massa                      Taroudant

                                  Projet Extension dessalement de Chtouka               576            300

                                  Projet Tan-Tan : Développement de l'agriculture
 Guelmim-Oued Noun                                                                     2.000           300
                                  à forte valeur ajoutée dans la région
                                        Total                                          19.701          1.300

    ► Programme de l'agriculture solidaire
La répartition régionale des projets de l’agriculture solidaire, prévus en 2025, se présente
comme suit :
                                                                                         CP 2025
              Région                        Nombre de projets
                                                                                          (MDH)
 Béni Mellal- Khénifra                               15                                    86,75
 Casablanca-Settat                                    7                                    34,44
 Dakhla-Oued Ed-Dahab                                 3                                       7,10
 Drâa-Tafilalet                                       8                                    54,73
 Fès-Meknès                                          11                                       81,71
 Guelmim-Oued Noun                                   4                                     54,38
 Laâyoune - Sakia El Hamra                           4                                     16,38
 Marrakech-Safi                                      20                                    138,75
 L'Oriental                                          15                                    103,88
 Rabat-Salé-Kénitra                                  11                                    49,37
 Sous-Massa                                           8                                    75,61
 Tanger-Tétouan-AI Hoceima                           18                                    151,29
              Total                                 124                                   854,40
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


    ► Programme de développement des filières de production

        - Industrie aqroalimentaire :
                                                                              Coût         CP 2025
              Région                              Projet
                                                                             (MDH)          (MDH)

 Tanger-Tétouan-AI Hoceima    Agropole de Loukkos                              137          25,6

 L’Oriental                   Centre d'innovation Agroalimentaire (CIA)        33            2,1

 Rabat-Salé-Kénitra           Agropole du Gharb                                103           21

 Marrakech-Safi               Agropole d’EI Haouz                             124,4         33.4


 Béni Mellal-Khénifra         Agropole de Tadla                                200           1,2

                                 Total                                       597,4          83,3

        - Infrastructures de commercialisation :
                                                                          Contribution *   CP 2025
              Région                              Projet
                                                                             (MDH)          (MDH)
                              Abattoir des viandes rouges et marchés
 Tanger-Tétouan-AI Hoceima                                                     60            12
                              aux bestiaux

                              Marché de gros de Berkane                         70           30
 L’Oriental
                              Abattoir des viandes rouges et marchés
                                                                                83           30
                              aux bestiaux

                              Marché de gros de Meknès                          70           30

 Fès-Meknès                   Marché de gros de Fès                            40            20

                              Abattoir des viandes rouges et marchés
                                                                               57,5          20
                              aux bestiaux
                              Abattoir des viandes rouges et marchés
 Rabat-Salé-Kénitra                                                             58           22
                              aux bestiaux

                              Marché de gros d'Essaouira                       20            20

 Marrakech-Safi               Souks hebdomadaires                               16            6

                              Abattoir des viandes rouges                       26            9

 Casablanca-Settat            Abattoir des viandes rouges                       12            3

                              Souks hebdomadaires                               8             8
 Souss-Massa
                              Abattoir des viandes rouges et marchés
                                                                                58           20
                              aux bestiaux
                              Marché de gros de Béni Mellal                    100           30
 Béni Mellal-Khénifra         Abattoir des viandes rouges et marchés
                                                                                21            3
                              aux bestiaux
                                                   NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


              Région                                    Projet                   Contribution *    CP 2025
                                      Abattoir des viandes rouges et marchés
 Drâa-Tafilalet                                                                            13         5
                                      aux bestiaux

 Guelmim-Oued Noun                    Abattoir des viandes rouges                      57,7          20

                                      Souks de Dakhla                                  80            20
 Dakhla Oued Ed-Dahab
                                      Abattoir des viandes rouges et marchés
                                                                                           56         18
                                      aux bestiaux
                                         Total                                       906,2           326
(*) Contribution du ministère de /'Agriculture


11.3.3.2 Programme de Réduction des Disparités Territoriales et Sociales
Dans la perspective de démarrer une deuxième phase de ce programme, les prévisions
budgétaires à moyen terme, couvrant la période 2025-2027, sont maintenues à hauteur des
contributions annuelles du budget de l’Etat, soit 1,1 MMDH, compte tenu du financement des
actions de partenariat local pour le développement du monde rural. Les crédits alloués au titre
de l’année 2025 se présentent comme suit :

                                                                                                  CP 2025
                                                 Programme
                                                                                                   (MDH)

  Programme de lutte contre les disparités territoriales et sociales dans le monde rural           1.000

 Partenariat avec les collectivités territoriales et les acteurs de la société civile agissant      100
 dans le domaine du développement rural

                                                   Total                                           1.100

11.3.3.3 Pêche maritime
La stratégie Halieutis a permis d’insuffler une nouvelle dynamique au secteur halieutique au
Maroc, notamment en ce qui concerne la durabilité de la ressource, la performance du secteur
et sa compétitivité. C’est dans ce cadre que s’inscrivent les efforts déployés par le
Gouvernement pour la consolidation des acquis, tout en assurant le développement du
secteur en synergie avec les grandes orientations stratégiques du pays.
C’est dans ce sens qu’un accord de coopération a été signé entre le Maroc et le Japon durant
l’année 2024, à l’effet de renforcer et de dynamiser la collaboration entre les deux pays dans
le domaine de la pêche maritime et de l’aquaculture. En effet, lors de la 38eme session de la
consultation annuelle Maroc-Japon sur la pêche, tenue à Rabat, les deux parties ont réaffirmé
leur engagement en faveur de la gestion durable des ressources halieutiques. Cet accord a
notamment intégré le projet de coopération technique "Aquaculture pour la croissance bleue
au Maroc", lancé en février 2024.
Ainsi, les projets d’investissement programmés au titre de l’année 2025 porteront sur :
    -    L’appui aux secteurs de la pêche et de l’aquaculture ;
    -    La construction des infrastructures administratives et de service ;
    -    Le financement des programmes et projets de recherche.
Ces projets sont présentés par région dans le tableau ci-après :
PROJET DE LO! DE FINANCES POUR L'ANNEE 2025



                                                                                                          Coût          CP 2025
           Région                                             Projet
                                                                                                         (MDH)           (MDH)
                                 Acquisition des sennes tournantes renforcées contre les
                                 attaques du grand dauphin en Méditerranée au profit                        90               13
                                 des armateurs
 Tanger-Tétouan-AI
                                 Mise en service et exploitation du centre de la mer à la
 Hoceima                                                                                                    65               10
                                 ville d’AI Hoceima

                                  Développement des projets d’économies bleues -
                                                                                                           59,3              10
                                  Projets aquacoles

                                 Acquisition des sennes tournantes renforcées contre les
                                 attaques du grand dauphin en Méditerranée au profit                        90               14
                                 des armateurs

 L’Oriental                      Construction et gestion d'un marché de gros de poisson
                                                                                                            45              22,5
                                 dans la ville de Béni Ensar - Nador

                                 Appui aux unités de valorisation des produits aquacoles
                                                                                                           127,2            30
                                 et aux unités liées à la production d'intrant

 Rabat-Salé -Kénitra              Promotion de l’activité aquacole                                         35,1             10,3

                                 Acquisition d'une vedette de sauvetage au niveau du
                                                                                                            30              30
                                 port de Casablanca
 Casablanca-Settat
                                  Renforcement des moyens de surveillance                      des
                                                                                                           69,5             52,4
                                  ressources halieutiques et de la biodiversité

 Souss-Massa                      Mise en œuvre d'un Aquapôle à Tiguert                                     38               4

                                 Acquisition d’une vedette au             niveau du port de
 Laâyoune-Sakia El Hamra                                                                                    30              30
                                 Laâyoune

                                 Aménagement de l'espace dédié à l'aquaculture dans la
                                                                                                            101             20,9
                                 zone de Boutalha - Région de Dakhla-Oued Ed-Dahab
 Dakhla-Oued Ed-Dahab
                                 Appui aux unités de valorisation des produits aquacoles
                                                                                                           127,2             20
                                 et aux unités liées à la production d'intrant

                                              Total                                                       907,3            267,1
(*) La réalisation de ces projets demeure tributaire de la signature des conventions y afférentes qui sont en cours d'élaboration.


11.3.3.4 Secteur de l’énergie
Le Maroc s'est fixé un objectif ambitieux d'atteindre plus de 52 % de sa capacité de
production d'électricité à partir de sources d'énergie renouvelables d'ici 2030. Cet
engagement s'inscrit dans une vision globale de transition énergétique, qui vise à réduire la
dépendance aux énergies fossiles et à diminuer l'empreinte carbone du pays.
A fin août 2024, le Maroc a déjà réalisé des avancées significatives, atteignant environ 44,3 %
de sa capacité de production d'électricité provenant de sources renouvelables. Ce progrès est
le fruit des investissements massifs dans des projets emblématiques tels que le complexe
solaire Noor à Ouarzazate, l'un des plus grands au monde, ainsi que divers parcs éoliens le
long de la côte atlantique.
                                                  NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


En 2025, cette stratégie se poursuivra avec un engagement renforcé pour les énergies
renouvelables. Le Gouvernement continuera d'investir dans des projets d'envergure, visant à
accroître sa capacité de production et à atteindre ses objectifs ambitieux en matière de
durabilité. De ce fait, l’année 2025 sera marquée par la réalisation des principaux projets
suivants :
     ► Projets en énergie solaire
                                                                          Capacité            Coût       CP 2025*
       Région                                   Projet
                                                                           (MW)              (MDH)        (MDH)

                       NOOR Midelt 1 suivant une configuration
                                                                              805            7.800        3.680
                       hybride CSP / PV

 Drâa-Tafilalet        Centrale solaire PV NOOR Midelt II                     390            4.631,9      2.263,8


                       Centrale solaire PV NOOR Midelt III                    390            4.631,9      2.263,8

                       NOOR Atlas : réalisation, exploitation et
                       maintenance de six centrales solaires
 Multirégional                                                                290            2.363,5       1.228
                       photovoltaïques en deux lots séparés, ainsi
                       que les lignes de raccordement y afférentes
                                   Total                                      1.875         19.427,3     9.435,6
(*) Ces projets ne sont pas budgétisés sur le Budget Général.

     ► Projets en énergie éolienne
                                                                                  Capacité       Coût    CP 2025*
       Région                                       Projet
                                                                                   (MW)         (MDH)     (MDH)

 Tanger-Tétouan-        Nassim Koudia Extension : construction et entretien
                                                                                      150       2.474      1.212,7
 Al Hoceima             de ce nouveau parc éolien


                        Construction et entretien du nouveau parc éolien
 Fès-Meknès                                                                           80         1.019    509,35
                        Nassim Taza II


 Laâyoune-Sakia         Construction et entretien du nouveau parc éolien
                                                                                      50         808      404,25
 El Hamra               Nassim Tarfaya Extension

                                        Total                                         280       4.301     2.126,3
(*) Ces projets ne sont pas budgétisés sur le Budget Général.


11.3.3.5 Artisanat
Conformément aux Hautes Orientations Royales et aux recommandations du Nouveau Modèle
de Développement, le Gouvernement a adopté la stratégie 2021-2030 visant à revitaliser le
secteur de l'artisanat. Cette stratégie est axée sur l'accompagnement des acteurs, la
structuration et la modernisation des filières, le renforcement des compétences des artisans et
l'amélioration du cadre sectoriel tout en favorisant l'inclusion sociale.
La dynamique engendrée par la nouvelle stratégie s'est enclenchée avec trois phases
majeures de développement :
PROJET DE LOI DE FINANCES POUR L'ANNEE 2025


    •    La phase de « Relance » (2021-2022), qui a marqué un tournant résultant de la forte
         impulsion de l'État pour lancer les premiers chantiers prioritaires ;
    •    La Phase de « Transformation » (2023-2025), actuellement en cours. Elle se focalise sur
         la consolidation des acquis tout en lançant de nouveaux chantiers structurants pour
         renforcer la compétitivité et l'innovation. En effet, cette transformation se positionne au
         cœur des efforts entrepris en 2024-2025, visant à moderniser les structures et à
         préparer la future accélération ;
    •    La phase « Accélération » (2025-2030) qui ciblera la capitalisation sur les progrès
         réalisés, pour accélérer les projets en cours et favoriser un développement durable et
         pérenne, consolidant ainsi la place de l'artisanat dans l'économie et dans la culture des
         marocains.
Les projets d’investissement programmés au titre de l’année 2025, s’inscrivent dans le cadre
du déploiement de la stratégie 2021-2030. Ils se présentent par région dans le tableau ci-
après :
                                                                                                Coût   CP 2025
        Région                                       Projet
                                                                                               (MDH)    (MDH)
                      Création d’un Centre de Formation par Apprentissage (CFA) à
 L’Oriental                                                                                     20        3
                      Driouch
                      Travaux d’extension et de réhabilitation de l’institut Spécialisé des
 Fès-Meknès           Arts Traditionnels de Fès, restructuré dans le cadre du MCC/              17,2      3
                      fonds «CHARAKA»

 Rabat-Salé-Kénitra   Construction de Dar Debagh à la commune de sidi Bouknadel                 44       4,9


                      Réalisation et équipement de la première tranche de la Zone
 Drâa-Tafilalet                                                                                 45,5      6
                      d’activité artisanale à Tarmigte


 Souss-Massa          Création d’un CFA à Agadir                                                50        5


 Toutes les régions   Normalisation et labellisation des produits et services de l'artisanat   40,5       8


                                          Total                                                217,2    29,9


11.3.3.6 Tourisme
Convaincu de la singularité de la destination Maroc et de sa capacité à rivaliser avec les
grandes destinations mondiales, le Maroc ambitionne d’accueillir 26 millions de touristes d’ici
2030, selon les objectifs de la feuille de route 2023-2026 lancée dernièrement. Cette stratégie
s’inscrit dans le cadre de la Vision Royale pour le développement du secteur et vise à stimuler
l’investissement et à favoriser l’emploi.
En 2024, le secteur du tourisme a connu une dynamique avec 11,8 millions de touristes et 18,7
millions de nuitées, enregistrés au titre de la période allant de janvier à août 2024, soit des
hausses respectives de 16% et 7% par rapport à la même période de l’année 2023.
Parallèlement, sur la même période de l’année 2024, les recettes voyages ont atteint 76,4
milliards de dirhams, soit une augmentation de +6,7% en glissement annuel.
                                                 NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT


La répartition régionale des principaux projets d’investissement dans le secteur du tourisme,
programmés pour l'année 2025, s'établit comme suit :
                                                                                                    Coût           CP 2025
       Région                                           Projet
                                                                                                   (MDH)            (MDH)

                        Mise à niveau des établissements de formation relevant du
 Rabat-Salé-            département du tourisme - repositionnement du centre de
                                                                                                      2                1,3
 Kénitra                qualification professionnelle hôtelière et touristique de
                        Touarga

                        Programme d’appui au développement de la TPME touristique
                                                                                                     564               131
                        « Programme Al Moukawala Siyahia »
 Multirégional
                        Appui à l’investissement pour le développement des filières
                                                                                                    1.637            512,5*
                        touristiques - Feuille de Route-

                                             Total                                                  2.203            644,8
(*) Les conventions avec les régions pour la réalisation des projets sont en cours d'élaboration. (Projets pilotés par la Société
Marocaine d'ingénierie Touristique (SMIT))
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025


Annexe : Répartition régionale des principaux projets d’investissement des Etablissements et Entreprises publics prévus au
titre de l’année 2025


                     Établissement ou                                                                                    Coût     Crédits 2025
  Région                                                                         Projet                                (en MDH)     (en MDH)
                     Entreprise public

                  Investissement 2025 de la Région Rabat - Salé - Kénitra : 18.995 MDH
                                                Construction de collèges                                                121,00       15,00

                    Académies Régionales        Construction, réhabilitation et équipement des salles du préscolaire   140,00        120,00
                      d'Education et de
                      Formation (AREF)          Construction des lycées                                                104,00        15,00

                                                Réhabilitation des écoles primaires                                    130,00        91,00
                    Agence Marocaine de
                    Développement de la         Développement de la zone logistique de Kénitra (45 Hectares)           250,00        45,00
                     Logistique (AMDL)
                                                Réhabilitation des infrastructures                                     820,43        114,49
Rabat-Salé-          Office National des
                                                Projets de Gares                                                       325,00        34,03
  Kénitra          Chemins de Fer (ONCF)
                                                Projets d’ateliers et centres de maintenance                            211,30       33,54
                   Fonds Hassan II pour le
                      Développement             Réalisation du Musée National de l'Archéologie et des sciences de la
                                                                                                                       900,00        200,00
                    Économique et Social        terre
                           (FDSHII)
                     Office National Des
                                                Construction d'un nouveau terminal à l'aéroport de Rabat               1.828,00      830,00
                     Aéroports (ONDA)
                                                Projets d'alimentation en Eau Potable de la zone urbaine               2.294,20      217,16
                       Office National de
                    l’Électricité et de l’Eau   Projets d'alimentation en Eau Potable de la zone rurale                459,64        69,10
                        Potable (ONEE)
                                                Projets d'assainissement                                                558,68       93,94
                                                                                    NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                Établissement ou                                                                                       Coût     Crédits 2025
  Région                                                                   Projet                                    (en MDH)     (en MDH)
                Entreprise public
                                           Contrat de maintenance longue durée pour la centrale TAG KENITRA           157,89       89,54

                                           Création de la nouvelle transformation 225/60 Kilovolts (kV) Khémisset     183,40       53,36
                                           Acquisition de 14 Autotransformateurs (ATR) de 400/225 kV de 450
                  Office National de                                                                                 749,00        32,90
                                           Mégavolts ampères (MVA)
               l’Électricité et de l’Eau
                   Potable (ONEE)          Poste 225/60 kV DAR GUEDDARI et son raccordement                           210,23       30,00

                                           Télé-relève des compteurs des clients BT et mise en place d’une
                                           Infrastructure de Comptage Avancée AMI (Advanced Metering                 300,00         4,01
                                           Infrastructure).

                                           Programme Extension des Périmètres d'irrigation (Aménagement de la
                                                                                                                      7.000        700,00
               Office Régional de Mise     zone sud-Est de la plaine du Gharb sur 30.000 ha)
                en Valeur Agricole du      Plan National d'économie d'eau en irrigation (Travaux de
                  Gharb (ORMVAG)           remplacement des conduites en amiante ciment par des conduites en          215,00       38,00
Rabat-Salé-                                PEHD au niveau des secteurs reconvertis en irrigation localisée)
  Kénitra        Office National des       Projet du Gazoduc Nigéria-Maroc (GNM)                                     476,50        67,00
                Hydrocarbures et des
                  Mines (ONHYM)            Travaux d'abandon des puits (Gharb)                                        100,22       20,00
                                           Travaux d'équipement en réseau d'assainissement liquide du grand
                 Régie Autonome de                                                                                   490,00        148,00
                                           bassin A - Tranche 1
                Distribution d’Eau et
               d’Électricité de Kénitra    Réalisation de la station de prétraitement des eaux usées industrielles   100,00        59,00
                        (RAK)              Réalisation de la station d'épuration des eaux usées (STEP) Kénitra
                                                                                                                     400,00        165,00
                                           Ouest
                                           Projet Zaitoune Tranche 1-A                                                154,91       30,00
                                           Projet Zaitoune Tranche 2-B                                                188,00       100,00
                                           Projet Rayhane Construction                                                135,60        38,20
              Holding Al Omrane (HAO)
                                           Projet Baie de Tamesna                                                     133,40       80,00
                                           Projet Relogement (AL Karama)                                              413,50       151,44
                                           Projet Sidi Taibi 2A2                                                      122,47        16,85
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                     Établissement ou                                                                                      Coût     Crédits 2025
  Région                                                                       Projet                                    (en MDH)     (en MDH)
                     Entreprise public
                                               Projet AL Mountazah Extension Tranche 3                                    124,36       16,09
                                               Projet AL Mountazah Extension Tranche 2                                    143,36       16,00
                                               Projet Mkhalif rest Tranche 2 à Kénitra (Zone de recasement)               172,25       11,00
                                               Projet AL Hadika 2                                                         368,13       80,00
                                               Projet EL Menzeh Bouregrag Tranche 3 FVIT (Nouvelle numérotation:
                                                                                                                          104,60       50,00
                  Holding Al Omrane (HAO)      tranche 4)
                                               Projet EL Menzeh Bouregrag Tranche 2 PROMo                                 119,55         -
                                               Projet Misk Lil                                                           164,80        60,87
                                               Projet Felline Tranche 2                                                  109,44        30,00
                                               Projet AL Wifak Extension                                                  111,00       10,00
                                               Projet Al Wahda Lotissement Tranche 4                                     249,64          -
                                               Projet Zone d'urbanisation nouvelle de Bouknadel                          2.700,00      300,00
                                               Projet Musée de l'archéologie                                             900,00        300,00

Rabat-Salé-                                    Convention de construction, extension et mise à            niveau   des
                                                                                                                          432,23       83,00
                                               établissements scolaires de la Région Rabat-Salé-Kénitra
  Kénitra
                    Société Rabat Région       Création d'un Hôpital de Psychiatrie de 240 lits dans la Commune de
                                                                                                                          197,00       150,00
                    Aménagement (SRRA)         Ain Aouda
                                               Création d'un Hôpital de Pneumologie et Phtisiologie de 120 lits au
                                                                                                                          212,00       150,00
                                               niveau de Tamesna
                                               Construction de la zone résidentielle dans le cadre du projet Cité
                                                                                                                         300,00        120,00
                                               artisanale Oulja Salé
                                               Reconstruction de la Cour de cassation                                    400,00        100,00
                                               Travaux de dédoublement de la route nationale 4 reliant Sidi Slimane à
                                               Sidi Kacem du Point Kilométrique (PK) 56+609 au PK 72+276 - Lot            129,60
                                               unique (Province de Sidi Kacem)
                        Caisse pour le
                     Financement Routier       Travaux de dédoublement de la route nationale 4 du PK 34+608 au PK                      217,15
                                                                                                                          127,28
                            (CFR)              51+197 (Lot 2) sur 16+589 KM (Province de Sidi Slimane)
                                               Travaux de dédoublement de la route nationale 4 du PK 18+800 au PK
                                                                                                                          120,30
                                               34+608 (lot 1) sur 15+808 KM (Province de Sidi Slimane).
                                                                                    NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



               Établissement ou                                                                                         Coût     Crédits 2025
  Région                                                                   Projet                                     (en MDH)     (en MDH)
               Entreprise public
             Investissement 2025 de la Région Tanger - Tétouan - Al Hoceima : 9.862 MDH
              Agence Marocaine des        Koudia Al Baida 1                                                           1.460,00         -
              Énergies Renouvelables
                      (MASEN)             Koudia Al Baida II                                                          2.425,50      1.212,75
              Académies Régionales        Construction de collèges                                                    104,00        26,00
                  d'Education et de
                 Formation (AREF)         Réhabilitation des écoles primaires                                         100,00        70,00
                 Société Marocaine
                                          Programme d’appui au développement de la TPME touristique               «
              d'ingénierie Touristique                                                                                564,00        131,00
                                          Programme Al Moukawala Siyahia »
                       (SMIT)
                                          Réhabilitation des infrastructures                                          1.359,59      206,31
               Office National des
                                          Projets Etudes et fonciers                                                  276,40         1,94
             Chemins de Fer (ONCF)
                                          Projets Bâtiments et Embranchements particuliers                             153,39       36,34
                Régie Autonome de         Réalisation de la Station d’Épuration pour la ville de Larache              220,00        52,00
 Tanger -       Distribution d’Eau et
 Tétouan      d’Électricité de Larache    Réalisation de la station d’épuration des eaux usées de Ksar El Kébir        187,00       53,00
Al Hoceima            (RADEEL)
                                          Projets d'alimentation en Eau Potable de la zone urbaine                    2.015,00      335,17
                                          Projets d'alimentation en Eau Potable de la zone rurale                     2.511,95      226,44

                                          Projets d'assainissement                                                     337,58       125,73

                                          Projet Centrale OCGT (Open Cycle Gas Turbine) Al Wahda                      4.499,84     1.798,94
                 Office National de
                                          Raccordement de la 2eme et 3eme tranche de la centrale CC (combined
              l’Électricité et de l’Eau                                                                               1.500,00      450,00
                                          cycle) Tahaddart
                  Potable (ONEE)
                                          Achat de 26 transformateurs 100 MVA et 150 MVA                              497,00         57,79

                                          Remplacement des supports basse tension (BT) endommagés                     223,48        40,87
                                          Télé-relève des compteurs des clients BT et mise en place d’une
                                          Infrastructure de Comptage Avancée AMI (Advanced Metering                   300,00          1,58
                                          Infrastructure).
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                             Établissement ou                                                                                     Coût      Crédits 2025
   Région                                                                             Projet                                    (en MDH)      (en MDH)
                             Entreprise public
                              Office National des
                                                      Investissement dans le cadre de la gestion et l'exploitation du Gazoduc
                             Hydrocarbures et des                                                                                 140,11        22,61
                                                      Maghreb-Europe (GME)
                               Mines (ONHYM)
                                                      Aménagement hydro agricole - Dar Khrofa                                   2.822,00        2,80
                            Office Régional de Mise   Programme National d'Economie d'Eau en Irrigation (PNEEI) - 2eme
                             en Valeur Agricole du                                                                               470,00         16,9
                                                      tranche
                              Loukkos (ORMVAL)
                                                      Projet d'interconnexion du barrage Oued El Makhazine avec le barrage
                                                                                                                                 821,00           -
                                                      Dar Khrofa
                                                      Projet de développement de l'aéroport de Tanger                           1.541,00       661,00
                              Office National des     Projet de développement de l’aéroport Tétouan Saniat R’mel -
                                                                                                                                 293,00        60,00
                              Aéroports (ONDA)        Nouveau Terminal
                                                      Projet de développement de l’aéroport Al Hoceima Acharif Al Idrissi        266,00           -
 Tanger -
 Tétouan                                              Lotissement Bled Talaroik Tranche 3                                        112,27         3,00
Al Hoceima                                            Projet Sidi Abid Tranche 3                                                 142,91         3,00

                           Holding Al Omrane (HAO)    Projet AL Wahda LS Tranche 2                                               194,56         8,00

                                                      Projet Ziriab                                                              116,60         35,21

                                                      Projet Chrafate Phase 1 -Tranche 2                                         232,60         8,00

                                                      Zone franche logistique & assiette foncière                               4.698,00       181,00

                             Tanger Med Spécial       Complexe portuaire                                                        29.721,00      311,00
                              Agency (TMSA)           Plateformes industrielles                                                 10.383,00      675,00

                                                      Projets pôle services                                                     2.010,00       473,00

                           Investissement 2025 de la Région Béni Mellal - Khénifra : 7.254 MDH
Béni Mellal -
                                                      Extension STEP Khouribga (+5.5 Mm3)                                        495,00         126,5
  Khénifra
  I \ l 1 kl* 1 11 1 1 U
                                   OOP SA
                                                      Station d'épuration des eaux usées (STEP) Fquih Ben Salah (5.5 Mm3)        737,00        181,38
                                                                                       NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                 Établissement ou                                                                                      Coût       Crédits 2025
  Région                                                                      Projet                                 (en MDH)       (en MDH)
                 Entreprise public
                                            Station d'épuration des eaux usées (STEP) Béni Mellal (9.5 Mm3)           494,00          63,15
                                            Station d'épuration des eaux usées (STEP) Kasba Tadla (2.2 Mm3)           304,00          10,00
                                            Mine Fokra                                                               4.100,00        1.526,00
                                            Laverie Fokra                                                            4.800,00        1.771,00
                        OCP SA
                                            Pipeline Secondaire Fokra                                                 700,00         256,00
                                            Utilités Fokra                                                            350,00          131,00
                                            Phase 1 - Centrales solaires Foum-Tizi / Ouled Fares (135 MW)             1.188,00        235,1
                                            Phase 2-Lot 1 Centrale solaire sidi Chennane (99 MW)                      778,95         445,35
                Office Régional de Mise
                 en Valeur Agricole du      Projet de résilience et durabilité de l'irrigation (REDI)                 750,00         254,00
                    Tadla (ORMVAT)
Béni Mellal -   Académies Régionales
                   d'Education et de        Mise à niveau des bâtiments du primaire et collégial                      208,55          96,46
  Khénifra
                   Formation (AREF)
                  Office National des
                                            Réhabilitation des infrastructures                                         142,67          5,58
                Chemins de Fer (ONCF)
                                            Projets d'alimentation en Eau Potable de la zone urbaine                  2.327,43        218,33

                                            Projets d'alimentation en Eau Potable de la zone rurale                   306,98          38,97

                                            Projets d'assainissement                                                   39,50          0,62
                   Office National de
                                            Electrification des villages dans le cadre du Programme de Réduction
                l'Electricité et de l'Eau                                                                             1.221,00        80,39
                                            des Disparités Territoriales et Sociales (PRDTS)
                    Potable (ONEE)
                                            Achat de 26 transformateurs 100 MVA et 150 MVA                            497,00          21,50

                                            Mise à niveau du poste 225/150/60KV Tizgui                                 43,50          21,49

                                            Réalisation de la transformation 225/60 KV à Mgila                         85,00          11,85
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                     Établissement ou                                                                                       Coût     Crédits 2025
   Région                                                                         Projet                                  (en MDH)     (en MDH)
                     Entreprise public
                       Office National de       Télé-relève des compteurs des clients BT et mise en place d’une
                    l'Electricité et de l'Eau   Infrastructure de Comptage Avancée AMI (Advanced Metering                 300,00         5,73
                        Potable (ONEE)          Infrastructure).

                                                Construction de la Station d'épuration et des collecteurs hors site des
                     Régie Autonome de          eaux usées et des eaux pluviales de l'Agropole de Béni Mellal et des       110,00         1,4
                    Distribution d'Eau et       centres avoisinants
Béni Mellal -       d'Electricité du Tadla
  Khénifra                (RADEET)              Raccordement du quartier Ouled Ayad et la Zone industrielle à la
                                                                                                                           142,00          -
                                                station d'épuration des eaux usées (STEP)

                                                Projet Pôle Urbain Diyar Al Mostakbal Tranche 1                            152,34       101,56

                  Holding Al Omrane (HAO)       Projet Pôle Urbain Diyar Al Mostakbal Tranche 2                            190,36       161,94

                                                Projet Pôle Urbain Diyar Al Mostakbal Tranche 3                            372,17       361,51

                  Investissement 2025 de la Région Fès - Meknès : 7.940 MDH
                     Régie Autonome de
                    Distribution d’Eau et       Construction de la nouvelle station de traitement à boues activées
                                                                                                                          580,00        360,00
                   d’Électricité de Meknès      moyenne charge - 1ere phase
                          (RADEM)
                    Académies Régionales        Réhabilitation des écoles primaires                                       120,00        80,00
Fpç - Mpl/npÇ         d'Education et de
                      Formation (AREF)          Création des lycées qualifiants                                            131,30       52,52
                    Agence Marocaine de
                                                Projet de développement de la Zone logistique à Fès de 32 Ha (Ras El
                    Développement de la                                                                                   300,00        50,00
                                                Ma)
                     Logistique (AMDL)
                     Régie Autonome de
                    Distribution d’Eau et       Réalisation de la station d'épuration par boues activées à chaîne
                                                                                                                           271,00       150,00
                    d’Electricité de TAZA       Tertiaire
                         (RADEETA)
                                                                                     NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                Établissement ou                                                                                     Coût       Crédits 2025
  Région                                                                    Projet                                 (en MDH)       (en MDH)
                Entreprise public
                                           Convention du programme complémentaire de mise en valeur de la
               Fonds Hassan II pour le                                                                              583,00          44,00
                                           Médina de Fès 2018-2023
                  développement
                économique et social       Convention du programme complémentaire de mise en valeur de la
                     (FDSHII)                                                                                       800,00         120,00
                                           Médina de Meknès 2018-2023
                Moroccan Agency for
                                           Projet Taza II                                                           1.018,71       509,35
               Solar Energy (MASEN)
                                           Réhabilitation des infrastructures                                        121,65         16,36

                 Office National des       Projets du Programme matériel roulant                                    362,05          110,29
               Chemins de Fer (ONCF)       Projets des Ateliers et centres de maintenance                           225,00          34,40
                                           Projets Gares                                                             177,27         68,60
                  Société Marocaine
               d'ingénierie Touristique    Développement du Tourisme culturel                                       147,00          9,00
                        (SMIT)
Fès - Meknès
                                           Projets d'alimentation en Eau Potable de la zone urbaine                 3.758,16       463,86

                                           Projets d'alimentation en Eau Potable de la zone rurale                  1.648,94       201,98
                                           Projets d'assainissement                                                 259,57          19,34
                                           Construction de la station d'épuration des eaux usées       (STEP) El
                                                                                                                   4.364,00        428,00
                                           Menzel
                  Office National de
               l'Electricité et de l'Eau   Raccordement de la Centrale TAG d’AI Wahda                               653,00         220,00
                   Potable (ONEE)
                                           Acquisition de 14 Autotransformateurs (ATR) 400/225 kV de 450 MVA        749,00          32,90

                                           Achat de 26 transformateurs 100 MVA et 150 MVA                           497,00          31,73

                                           Télé-relève des compteurs des clients BT et mise en place d’une
                                           Infrastructure de Comptage Avancée AMI (Advanced Metering                300,00          6,96
                                           Infrastructure).
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                     Établissement ou                                                                                     Coût     Crédits 2025
  Région                                                                        Projet                                  (en MDH)     (en MDH)
                     Entreprise public
                     Office National Des
                                               Projet de développement de l'aéroport Fès Saiss - Nouveau Terminal       500,00        480,00
                     Aéroports (ONDA)
                                               Projet Station de traitement des boues de la station d'épuration des
                                                                                                                         133,00         -
                     Régie Autonome de         eaux usées (STEP)
                     Distribution d'Eau et     Projet Complexe Himmer / Miyit                                            126,00        5,74
                     d'Electricité de Fès
                           (RADEEF)            Projet Collecteur principal Oued Fès                                     333,00        100,00
                                               Réutilisation des eaux usées de la ville de Fès                           310,00       30,00
                                               Programme d'aménagement des Parkings, des espaces et d'installation
                                                                                                                        300,00          -
                                               d'un dispositif d'information dans la Médina de Fès (117)

                       Agence pour le          Programme complémentaire de la mise en valeur de la Médina de Fès -
                                                                                                                        583,00          -
                     Développement et la       Convention Mai 2018-
                     Réhabilitation de la      Programme de réhabilitation et de valorisation de l'ancienne Médina de
                                                                                                                        800,00          -
                    Médina de Fès (ADER)       Meknès 2019-2023 - Convention Octobre 2018
Fès - Meknès                                   Programme de Mise en Valeur des Activités Économiques et
                                               d’Amélioration du Cadre de Vie dans l’Ancienne Médina de Fès (2020-      670,00          -
                                               2024)
                                               Projet Yakout 3 Tranche 1                                                 189,37       26,00

                                               Projet Jnane Azzaitoune Tranche 1                                         215,59       15,00

                                               Projet Jnane Azzaitoune Tranche 2                                         199,42       20,00

                                               Projet Riad Saiss Tranche 1                                              403,00        20,00
                  Holding Al Omrane (HAO)
                                               Projet Riad Saiss Tranche 2                                               275,32       20,00

                                               Projet Zone Industrielle d’Ouislane                                      244,00        45,00
                                               Projet station d'épuration des eaux usées-Parc industriel ain cheggag
                                                                                                                         126,26       45,00
                                               (STEP-PIAC)
                                               Projet mise à niveau urbaine complémentaire Meknès                        142,00       30,00
                                                                                   NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                 Établissement ou                                                                                           Coût     Crédits 2025
  Région                                                                  Projet                                          (en MDH)     (en MDH)
                 Entreprise public

                                         Projet mise à niveau urbaine (MANAU) Fès tranche 2                               200,00        50,00

                                         Projet mise à niveau urbaine Missour                                              161,00       30,00
                                         Projet mise à niveau urbaine Taza tranche 2                                       196,00       19,60
                                         Création  du   fonds de      renforcement        des       accessibilités   et
                                                                                                                          500,00        20,00
                                         désenclavement rural (accessibilité urbaine)
                                         Programme Régional de Mise à Niveau des Centres Emergents                         244,14       25,00

                                         Programme régional de Mise à Niveau des Centres Urbains                          455,00        25,00
Fès - Meknès   Holding Al Omrane (HAO)
                                         Création de locaux industriels (lutte contre l'informel)                         300,00        25,00

                                         Création d'une zone industrielle d’Oued Amlil pour la valorisation des
                                                                                                                          200,00        25,00
                                         matériaux de construction
                                         Création d'une zone Industrielle à Taounate                                       100,00       10,00

                                         Contribution à la mise en œuvre du Programme du développement
                                                                                                                          250,00        15,00
                                         urbain (PDU)

                                         Construction d'une nouvelle Gare routière à Meknès                                100,00        5,00

               Investissement 2025 de la Région Casablanca - Settat : 44.012 MDH
                                         Dessalement Plan d'urgence globale (60 Mm3/an)                                   2.297,00      66,05

Casablanca -                             Dessalement Phase d'extension wave II (60 Mm3/an)                                2.520,00      221,58
   Settat              OOP SA            Dessalement Casablanca (60 Mm3/an)                                               2.550,00      780,00
                                         Dessalement wave III                                                             5.324,00     2.400,00
                                         Projet Pipeline Jorf- Khouribga (80 Mm3/an)                                      5.000,00      482,62
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                     Établissement ou                                                                                    Coût     Crédits 2025
  Région                                                                            Projet                             (en MDH)     (en MDH)
                     Entreprise public
                                               Projet Pompage d'eau de mer Jorf new intake                             2.500,00     2.170,00
                                               Plateforme Green H2A et Pilote de Green Ammonia                          591,50       30,00
                                               Port Jorf Lasfar                                                        3.241,00         -
                                               Projet adaptation des installations de l'atelier phosphorique pour la
                                                                                                                       2.547,00         -
                                               pulpe et Line E
                                               Pompage eau de mer + Distribution d’eau de mer                          863,00           -

                                               Extension du Stockage de soufre                                          164,00       34,23
                                               Réhabilitation des installations Jorf Phosphate Hub (JPH)               1.833,50      120,00
                                               Construction lignes sulfurique D et U avec centrale intégrée - Jorf     4.892,00      356,53
                                               Ligne phosphorique F                                                    829,00        10,92
                                               OSBL (Outside Battery Limits) Ligne F (Ex Racks et tuyauterie
Casablanca -                                                                                                           434,30         21,32
                            OCP SA             Vaguelette)
   Settat                                      3 Nouvelles Lignes d'engrais TSPS                                       6.096,00      788,96
                                               Convoyeur Engrais Axe 3 / Nouveaux hangars de stockage du
                                                                                                                       1.137,00      180,93
                                               Phosphate de diammonium (DAP)
                                               Alimentation électricité JPH 2                                           171,00        4,00
                                               Racks et tuyauterie JPH 2                                               590,70        51,00
                                               Unité de stockage d'Ammoniac                                            579,00        219,02
                                               Construction d'une nouvelle ligne SAP 9 (Ligne E)                       1.638,00      243,48
                                               Extension nouvelle Centrale (Lignes SAP U & E)                          1.160,00      553,39
                                               Terrassements & VRD Phase 2                                             100,00        18,66
                                               Projet Amélioration et flexibilité                                       155,00          -

                                               Unités de décadmiation d'acide 1 x 1500 T/j                             274,00        20,00
                                                                              NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



               Établissement ou                                                                               Coût       Crédits 2025
  Région                                                             Projet                                 (en MDH)       (en MDH)
               Entreprise public
                                   Construction d’une nouvelle ligne Sulfurique 1,7 Mt Jorf                 2.500,00       1.000,00
                                   Extension poste électrique                                                345,00         120,00
                                   Construction d'une unité de fusion filtration 263 bis                     1.100,00       343,30
                                   Etude de production du MAP (Monoammonium Phosphate) Soluble
                                                                                                             1.593,00       800,00
                                   pour 4 unités 100KT

                                   Décadmiation d'acide sur résine                                           1.430,00        47,00

                                   Unité de décadmiation d'acide Procédé JACOBS 1 x 500 KT                   160,00          110,00
                                   Généralisation de la décadmiation d'acide à Jorf Lasfar                  1.600,00        300,00
                                   PPA unit (Projet Azur)                                                    1.945,00      1.000,00
                                   DCP/MCP (Phosphate Dicalcique/Phosphate Monocalcique) Animal
                                                                                                             600,00         200,00
Casablanca -                       feeds
                    OCP SA
   Settat                          MAP Crystal                                                               1.047,00       160,00
                                   DAP (Di-Ammonium Phosphate) Improvement                                   100,00          27,00
                                   Adaptation NPK (Azote, Phosphore, Potassium)                              200,00         103,00
                                   Unités TSP (Triple Superphosphate) 2 x 500 KT                             1.732,00       634,00
                                   Stockage de phosphogypse pour deux JFC                                    2.139,00       869,25
                                   Générateur à air chaud                                                    130,00          68,00
                                   Etudes Refroidissement eau de mer                                         180,00          54,00
                                   Construction Ligne TSP 2 MT Phase II                                      3.100,00       1.240,00
                                   Construction des nouveaux échelons CAP                                    700,00         420,00
                                   Programme Energie renouvelable (répartis entre régions de Casa, Béni
                                                                                                            15.800,00      3.075,00
                                   mellal et Marrakech)
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                     Établissement ou                                                                                     Coût     Crédits 2025
  Région                                                                        Projet                                  (en MDH)     (en MDH)
                     Entreprise public
                                               ODI (Owner Direct Investment) + JFC 1 (Jorf Fertilizers Company)         5.777,00      86,75
                                               ODI (Owner Direct Investment) + JFC 2 (Jorf Fertilizers Company)         5.639,00      93,26
                            OCP SA
                                               ODI (Owner Direct Investment) + JFC 3 (Jorf Fertilizers Company)         5.700,00         -

                                               ODI (Owner Direct Investment) + JFC 4 (Jorf Fertilizers Company)         5.599,00         -

                                               Elargissement à 2*3 voies de l'autoroute Casa - Berrechid                776,00        30,00
                                               Elargissement à 2*3 voies de l'autoroute de Contournement de
                                                                                                                        745,00        40,00
                                               Casablanca
                    Autoroutes du Maroc
                           (ADM)               Projet Autoroute Tit Mellil - Berrechid                                  2.500,00      1.128,19
                                               Projet Aménagement des nœuds d’Ain Harrouda et Sidi Maarouf              1.100,00      500,00
                                               Projet Automatisation                                                     371,43        61,54
                                               Travaux de construction du siège de l'ANP                                200,00        200,00
Casablanca -        Agence Nationale des
   Settat               Ports (ANP)            Travaux de renforcement des digues de protection            au port de
                                                                                                                        100,00        90,00
                                               Casablanca
                    Académies Régionales       Construction, réhabilitation et équipement du préscolaire                 145,00       120,00
                      d'Education et de
                      Formation (AREF)         Réhabilitation des écoles primaires                                       122,00       70,00

                    Agence Marocaine de
                                               Développement de la Zone Logistique au Sud de Casablanca de 70 Ha
                    Développement de la                                                                                 550,00        100,00
                                               (Oulad Saleh)
                     Logistique (AMDL)

                   Fonds Hassan II pour le
                      développement            Programme de traitement des constructions menaçant ruine dans
                                                                                                                        300,00        80,00
                    économique et social       l'ancienne médina de Casablanca
                         (FDSHII)

                     Office National des       Réhabilitation des infrastructures                                        152,89        4,25
                   Chemins de Fer (ONCF)       Projet Signalisation                                                      298,20        52,73
                                                                                    NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                Établissement ou                                                                                    Coût       Crédits 2025
  Région                                                                   Projet                                 (en MDH)       (en MDH)
                Entreprise public
                                           Projets Études et fonciers                                              103,63          0,62
                                           Projets Programme matériel roulant                                      668,03         107,72

                 Office National des       Équipement, Outillage & Système d'information                            84,00          5,00
               Chemins de Fer (ONCF)       Projets Ateliers et centres de maintenance                              1.014,16       370,30
                                           Projets Bâtiments et Embranchements particuliers                        226,90          54,90

                                           Projets Gares                                                           130,69          30,26
                                           Projets d'alimentation en Eau Potable de la zone urbaine                3.580,10       368,69
                                           Projets d'alimentation en Eau Potable de la zone rurale                 459,26          23,68
                                           Projets d'assainissement                                                 122,56          0,19
                                           Renforcement du réseau 400 kv Centre- Casablanca                        635,65         284,68
Casablanca -
   Settat         Office National de       Contrat de maintenance longue durée pour la centrale turbines à Gaz
                                                                                                                    351,65         90,00
               l'Electricité et de l'Eau   Mohammedia 2
                   Potable (ONEE)          Construction du Poste 225/60 kV et 225/22 kV de Benslimane II et son
                                                                                                                   250,40          74,20
                                           raccordement
                                           Achat de 26 transformateurs 100 MVA et 150 MVA                          497,00          60,28
                                           Télé-relève des compteurs des clients BT et mise en place d’une
                                           Infrastructure de Comptage Avancée AMI (Advanced Metering               300,00           5,91
                                           Infrastructure).
                                           Projet de développement de l'aéroport Mohammed V de Casablanca         12.000,00      1.000,00
                                           Projet "Quick Wins - Court Terme" de l'aéroport Mohammed V de
                                                                                                                   350,00          90,00
                 Office National Des       Casablanca
                 Aéroports (ONDA)          Projet "Actions d’harmonisation" de l'aéroport Mohammed V de
                                                                                                                   220,00         220,00
                                           Casablanca
                                           Mise à niveau de l'infrastructure de l’aéroport de Benslimane           350,00         305,00
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                     Établissement ou                                                                                          Coût     Crédits 2025
  Région                                                                         Projet                                      (en MDH)     (en MDH)
                     Entreprise public
                                                Ligne 2 du Tramway de Casablanca                                             4.280,71      72,00
                    Casablanca Transports       Ligne 3 et 4 du Tramway de Casablanca                                        7.036,60      689,00
                                                Ligne 5 et 6 du Bus à Haut Niveau de Service (BHNS) de Casablanca            1.875,00      45,00
                                                Extension et transformation du procédé d'épuration de la station
                                                                                                                              163,00        3,00
                                                d'épuration des eaux usées (STEP) de Deroua en boues activées
                                                Extension et transformation du procédé d'épuration de la station
                                                d'épuration des eaux usées (STEP) de Settat en boues activées - Filière      247,00         0,10
                     Régie Autonome de          domestique
                     Distribution d'Eau et      Extension et transformation du procédé d'épuration de la station
                  d'Electricité de la Chaouia                                                                                 118,00        0,10
                                                d'épuration des eaux usées (STEP) de Sidi Rahal plage
                           (RADEEC)
                                                Réalisation de la deuxième filière d'épuration domestique de la station
                                                                                                                             265,00          -
                                                d'épuration des eaux usées (STEP) de Had Soualem en boues activées
                                                Réalisation d'une station d'épuration en boues activées à la ville de
                                                Settat - filière industrielle, y compris la conduite de transfert des eaux   120,00         0,10
Casablanca -                                    usées industrielles
   Settat                                       Projet de basculement de l'alimentation en eau potable des zones
                                                desservies par le canal principal bas service vers le canal principal haut    127,00         -
                     Régie Autonome de          service dans la province de Sidi Bennour
                     Distribution d'Eau et
                                                Étude et réalisation d'une station d'épuration à Azemmour                     122,50       35,20
                   d'Electricité d'EI Jadida
                           (RADEEJ)             Construction     et    équipement       d'une     station  d'épuration
                                                commune      pour la     ville de     Bir jdid    et centre   M'Harza         125,00       20,00
                                                Sahel y compris acquisition du terrain
                                                Projet Grand Théâtre de Casablanca                                           1.440,00       1,07
                                                Réaménagement du Zoo d'Aïn Sebaa                                             250,00          -
                          Société               Projet de Trémie des Almohades                                               860,00         2,78
                     Casa-Aménagement
                           (SCA)                Aménagement de la Forêt de Bouskoura Merchich                                 110,00         -

                                                Aménagement de la Corniche d'Aïn Sebaa                                       100,00          -

                                                Aménagement de la Corniche de Mohammedia                                      125,00         -
                                                                              NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



               Établissement ou                                                                               Coût       Crédits 2025
  Région                                                             Projet                                 (en MDH)       (en MDH)
               Entreprise public
                                   Mise à niveau des voiries à l'intérieur de la ville de Casablanca         1.950,00          -

                                   Aménagement de la Route Régionale 322 (Point Kilométrique 5-Point
                                                                                                              161,00           -
                                   Kilométrique 10)

                                   Siège Régional de la Gendarmerie Royale (SRGR)                            203,00          31,59

                                   Entretien & maintenance des ouvrages d'art de Casablanca                  147,00            -

                                   Développement axe Jeunesse & Sports                                       204,00          54,59

                                   Aménagement des zones d'activités économiques dédiées aux unités
                                                                                                             500,00            -
                                   identifiées à risque

                    Société        Programme de mise à niveau & équipement des établissements dédiés
                                                                                                             120,00          19,41
               Casa-Aménagement    aux jeunes au niveau de la Région de Casablanca-Settat
Casablanca -         (SCA)
                                   Ouverture et aménagement des voiries au niveau de la Préfecture
   Settat                                                                                                    300,00          73,39
                                   d’arrondissement de Sidi Bernoussi

                                   Aménagement et dédoublement de la route régionale côtière RR 322,
                                   reliant les communes de Cherrat, Bouznika et El Mansouria, dans la        350,00            -
                                   Province de Benslimane
                                   Programme de développement des Infrastructures de mobilité Zone
                                                                                                            2.000,00        500,00
                                   Casa-Ouest
                                   Aménagement de voirie à l'intérieur de la Ville de Mohammedia             500,00         100,00

                                   Ouverture et Aménagement des voiries au niveau de la Préfecture
                                                                                                             120,00          6,00
                                   d’arrondissement Hay Hassani

                                   Projet Véhicule d'investissement                                         800,00          80,00
               ITHMAR AL MAWARID
                                   Projet Wessal Casa Port                                                  6.796,00
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                     Établissement ou                                                                                  Coût     Crédits 2025
  Région                                                                       Projet                                (en MDH)     (en MDH)
                     Entreprise public
                                               Extension de front d'accostage du quai 16 au terminal polyvalent au                   -
                                                                                                                      147,00
                                               port Jorf Lasfar
Casablanca -         Tanger Med Spécial
                      Agency (TMSA)            Approfondissement de la série 30 (240 ml) au port de Casablanca        475,50       110,40
   Settat
                                               Renouvellement de 2 portiques à conteneur au port de Casablanca       240,00        240,00

                  Investissement 2025 de la Région Marrakech - Safi : 19.648 MDH
                                               Dessalement Plan d'urgence globale (50 Mm3/an)                        3.304,00      306,00

                                               Projet Dessalement wave II (100 Mm3)                                  4.040,00      544,67
                                               Projet Pipeline Safi-Mzinda Phosphate Hub - Marrakech (2 x 100
                                                                                                                     6.000,00     2.000,00
                                               Mm3/an)
                                               Projet station d'épuration des eaux usées Safi (11.7 Mm3/an)          653,00        25,72
                                               Projet station d'épuration des eaux usées Marrakech (16 Mm3/an)       1.000,00      167,70
                                               Projet Pompage eau de mer                                             1.900,00     1.140,00
Marrakech -                                    Projet Dessalement wave III (60 Mm3)                                  2.100,00      890,00
   Safi                     OCP SA             Laverie Benguerir                                                     2.596,00      46,45

                                               Mine de M’zinda (Phase 1)                                              751,00         -

                                               Mine de Benguerir (Phase 1)                                           354,00          -

                                               Extension Mine de Benguérir                                           5.471,00     1.823,00

                                               Ouverture nouvelle Mine de Louta                                      7.018,00     2.083,00

                                               Usine de séchage à Mzinda                                             1.650,00      667,00

                                               Usines de production d'engrais à Safi                                 6.006,00     1.630,00
                                                                                    NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



               Établissement ou                                                                                     Coût       Crédits 2025
  Région                                                                  Projet                                  (en MDH)       (en MDH)
               Entreprise public
                                          Nouveau Port de Safi                                                    6.305,00        1.364,00

                                          Phase 1 - Centrales solaires Benguerir (67 MW)                           589,00          117,00

                                          Phase 2-Lot 1 centrale solaire (30 MW)                                   436,04          254,3
                      OCP SA
                                          Phase 2-Lot 2 centrale solaire (271 MW)                                  3.115,00       869,89

                                          Ligne sulfurique Safi (PS4)                                              1.449,00        45,19

                                          Stockage de phosphogypse à Safi                                         4.500,00         113,00

              Office Régional de Mise     Programme national d'économie d'eau en irrigation (PNEEI)               3.643,00        207,79
               en Valeur Agricole du
                 Haouz (ORMVAH)           Agriculture solidaire                                                    423,20          23,08
Marrakech -
                                          Projets d'alimentation en Eau Potable de la zone urbaine                 3.306,28        261,82
   Safi
                                          Projets d'alimentation en Eau Potable de la zone rurale                  790,29          80,17

                                          Projets d'assainissement                                                  217,90          6,72

                                          Acquisition de 14 Autotransformateurs (ATR) 400/225 kV de 450 MVA        749,00          65,80
                 Office National de
              l'Electricité et de l'Eau   Transformation 400/225 kV au poste TENSIFT II                            1.220,20        61,20
                  Potable (ONEE)
                                          Achat de 26 transformateurs 100 MVA et 150 MVA                           497,00          27,70
                                          Construction d'un poste 225/60/22KV CHRIFIA et Rabattement 60 &
                                                                                                                    217,87         12,53
                                          22 KV
                                          Télé-relève des compteurs des clients BT et mise en place d’une
                                          Infrastructure de Comptage Avancée AMI (Advanced Metering                300,00           4,26
                                          Infrastructure).
 PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                     Établissement ou                                                                                             Coût     Crédits 2025
  Région                                                                          Projet                                        (en MDH)     (en MDH)
                     Entreprise public
                    Agence Marocaine de
                                               Développement     de   la   Zone     logistique   à   Marrakech   de   37   Ha
                    Développement de la                                                                                         285,00        102,90
                                               (Tamensourt)
                     Logistique (AMDL)
                                               Construction des lycées collégiaux                                                115,84       39,87
                    Académies Régionales       Construction des lycées                                                           119,50       40,68
                      d'Education et de
                      Formation (AREF)         Construction, réhabilitation et équipement des salles du préscolaire             194,00        150,00

                                               Réhabilitation des écoles primaires                                              120,00        80,00
                     Office National Des
                                               Projet de développement de l'aéroport de Marrakech Ménara                        1.846,00      861,00
                     Aéroports (ONDA)

                  Holding Al Omrane (HAO)      Projet Tamouziga 1                                                                104,26       50,00

                                               Alimentation en eau potable des zones rurales de la Province de Safi à
                     Régie Autonome de                                                                                          334,00        164,00
                                               partir de la station de dessalement de l'OCP de Safi
Marrakech -          Distribution d'Eau et
   Safi              d'Electricité de Safi     Réalisation du projet d'assainissement des villes de jemâa shaim, sebt
                           (RADEES)                                                                                              188,00       60,00
                                               gzoula et centre de Tlet bouguedra

                    Agence Nationale des       Reconstitution et mise à niveau générale des régions touchées par le
                                                                                                                                960,70        112,61
                    Eaux et Forêts (ANEF)      séisme d'AI Haouz

                                               Réhabilitation des infrastructures                                                413,16        69,12
                     Office National des       Projet Signalisation                                                             605,00           -
                   Chemins de Fer (ONCF)
                                               Exploitation ferroviaire : Ateliers et centres de maintenance                    200,00        50,00

                                               Projet de sécurisation du réseau d'eau potable de la ville de Marrakech           285,65       30,00
                     Régie Autonome de
                     Distribution d'Eau et     Renforcement de la ligne Haute Tension (HT) du poste source
                                                                                                                                130,00        40,00
                  d’Electricité de Marrakech   Marrakech ville
                         (RADEEMA)
                                               Normalisation du poste Ville                                                     100,00        30,00
                                                                                    NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                Établissement ou                                                                                     Coût      Crédits 2025
  Région                                                                   Projet                                  (en MDH)      (en MDH)
                Entreprise public
                 Régie Autonome de         Renouvellement Réseau Eau potable                                        171,83         50,00
Marrakech -      Distribution d'Eau et
   Safi       d’Electricité de Marrakech   Réhabilitation du réseau d'assainissement                                380,00         60,00
                     (RADEEMA)
              Investissement 2024 de la Région de l’Oriental : 6.503 MDH
               ITHMAR AL MAWARID           Projet Société de développement de Saidia (SDS)                         3 225,00        8,50

              Nador West Med (NWM)         Projets Nador West Med                                                  12.639,00      826,00
               Fonds Hassan II pour le
                  développement
                                           Fonds d'investissement de la région de l'Oriental                        300,00         13,00
                économique et social
                     (FDSHII)
                                           Projets d'alimentation en Eau Potable de la zone urbaine                1.980,17       454,75

                                           Projets d'alimentation en Eau Potable de la zone rurale                  744,32         37,33

L’Oriental                                 Projets d'assainissement                                                 866,85         17,70

                                           Raccordement du poste 225 Kv du port Nador West Med                      270,52         86,55

                                           Évacuation de l’énergie électrique produite par la centrale à charbon
                  Office National de                                                                                688,74         70,40
               l'Electricité et de l'Eau   de Jerada
                   Potable (ONEE)
                                           Acquisition de 14 Autotransformateurs (ATR) 400/225 kV de 450 MVA        749,00         32,90

                                           Equipement des lignes 60 kV électrique par OPGW (Optical Ground
                                                                                                                    158,90         11,91
                                           Wire) sous tension électrique

                                           Télé-relève des compteurs des clients BT et mise en place d’une
                                           Infrastructure de Comptage Avancée AMI (Advanced Metering                300,00          11,11
                                           Infrastructure).
  PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                      Établissement ou                                                                                        Coût     Crédits 2025
   Région                                                                         Projet                                    (en MDH)     (en MDH)
                      Entreprise public
                                                 Pôle Urbain AL Yassamine                                                    497,77       127,10
                                                 Pôle Urbain Hamria                                                         539,60        179,34

                   Holding Al Omrane (HAO)       Lotissement Jnane Moulouya                                                  381,85       132,44
                                                 Pôle Urbain AL Arruit Tranche 3-4                                           334,75       45,88

                                                 Projet les jardins d'isly                                                   141,44        21,62

                    Office Régional de Mise      Programme national d'économie d'eau en irrigation : Projet de
                                                                                                                            870,00        342,21
                    en Valeur Agricole de la     modernisation des secteurs d’irrigation de la plaine du Garet
                     Moulouya (ORMVAM)           Plan d'urgence au niveau du bassin hydraulique de la Moulouya              900,00        240,00
 L’Oriental
                                                 Réhabilitation des infrastructures                                          103,25       27,96
                      Office National des
                                                 Projet Signalisation                                                        103,45       15,44
                    Chemins de Fer (ONCF)
                                                 Projets du Programme matériel roulant                                       276,26       157,32
                      Régie Autonome de
                     Distribution d'Eau et       Réutilisation des eaux usées traitées de la station d'épuration des eaux
                                                                                                                             218,00       10,00
                     d'Electricité d'Oujda       usées (STEP) d’Oujda pour l’arrosage des espaces verts
                           (RADEEO)
                      Office National des
                                                 Développement de la découverte de Tendrara via un projet micro-GNL
                     Hydrocarbures et des                                                                                    120,44       50,00
                                                 (Gaz Naturel Liquéfié)
                       Mines (ONHYM)

                   Investissement 2025 de la Région Drâa - Tafilalet : 8.495 MDH

                                                 Projets d'alimentation en Eau Potable de la zone urbaine                   1.589,10      211,39
Drâa-Tafilalet          Office National de
                     l'Electricité et de l'Eau   Projets d'alimentation en Eau Potable de la zone rurale                     736,41       72,46
                         Potable (ONEE)
                                                 Projets d'assainissement                                                    315,74       37,90
                                                                                       NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                   Établissement ou                                                                                        Coût     Crédits 2025
   Région                                                                     Projet                                     (en MDH)     (en MDH)
                   Entreprise public
                                              Electrification des villages dans le cadre du programme              de
                                                                                                                         2.256,00      79,40
                                              parachèvement PERG (Programme d'Electrification Rurale Global)

                                              Evacuation de la Centrale solaire de Midelt                                 553,26       67,33
                     Office National de       Construction   d'un   poste source    60/22KV   BLEIDA (Province de
                  l'Electricité et de l'Eau                                                                               33,13         8,80
                                              ZAGORA)
                      Potable (ONEE)
                                              Injection de nouveaux postes Moyenne Tension /Basse Tension au
                                                                                                                           9,14         8,74
                                              niveau du réseau électrique de la région
                                              Télé-relève des compteurs des clients BT et la mise en place d’une
                                              Infrastructure de Comptage Avancée AMI (Advanced Metering                  300,00         2,95
                                              Infrastructure).
                  Académies Régionales        Construction des collèges                                                   101,00       46,97
                    d'Education et de
Drâa-Tafilalet      Formation (AREF)          Réhabilitation des écoles primaires                                         177,78       82,67

                  Office Régional de Mise     Programme de développement des parcours et mise en œuvre de la loi
                   en Valeur Agricole de      n° 113-13 relative à la transhumance pastorale et l'aménagement et la      238,90        25,60
                  Ouarzazate (ORMVAO)         gestion des espaces pastoraux et sylvo-pastoraux

                  Office Régional de Mise     Projet de développement de l'irrigation et d'adaptation de l'agriculture
                   en Valeur Agricole du      irriguée aux changements climatiques à l'aval du barrage Kaddoussa         1.033,00      37,00
                  Tafilalet (ORMVATAF)        (PDIAAI-CC)

                                              Lotissement Jardins de Tafilalet Tranche 1                                 220,00        75,00

                                              Lotissement Jardins de Tafilalet Tranche 2                                  111,71        6,00
                 Holding Al Omrane (HAO)
                                              Lotissement Jardins de Tafilalet Tranche 3                                  125,00        7,00

                                              Lotissement Jardins de Tafilalet Tranche 4                                  221,00       10,00
  PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                      Établissement ou                                                                                 Coût     Crédits 2025
   Région                                                                        Projet                              (en MDH)     (en MDH)
                      Entreprise public
                                                 Projet Raccordement Noor Midelt (phase 1)                           544,23           -
                     Moroccan Agency for
Drâa-Tafilalet       Solar Energy (MASEN)
                                                 Projet Raccordement Noor Midelt (phase 2)                           600,00        199,98

                                                 Projet Noor Atlas                                                   768,00        614,00
                   Investissement 2025 de la Région Souss - Massa : 5.126 MDH
                      Office National Des
                                                 Projet de développement de l'aéroport d'Agadir Al Massira           1.530,00      600,00
                      Aéroports (ONDA)
                     Agence Marocaine de
                     Développement de la         Développement de la Zone logistique d'Agadir                        350,00         45,9
                      Logistique (AMDL)

                                                 Construction des écoles primaires                                   150,00        20,50
                     Académies Régionales
                       d'Education et de
                       Formation (AREF)          Réhabilitation des écoles primaires                                 150,00        50,36

                       Société Marocaine
                    d'ingénierie Touristique     Projet du site touristique d’Aghroud                                600,00        150,00
Souss - Massa                (SMIT)
                                                 Projets d'alimentation en Eau Potable de la zone urbaine            1.908,06      419,36
                                                 Projets d'alimentation en Eau Potable de la zone rurale             1.028,08      163,55

                                                 Projets d'assainissement                                            464,82         1,52

                        Office National de       Renforcement du Réseau 400 kV Sud (Lignes)                          998,60        73,87
                     l'Electricité et de l'Eau
                         Potable (ONEE)          Renforcement du Réseau 400 kV Sud (Postes)                           435,22        17,42

                                                 Acquisition de 14 Autotransformateurs (ATR) 400/225 kV de 450 MVA   749,00        16,45
                                                 Télé-relève des compteurs des clients BT et mise en place d’une
                                                 Infrastructure de Comptage Avancée AMI (Advanced Metering           300,00         14,77
                                                 Infrastructure).
                                                                                        NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                  Établissement ou                                                                                      Coût       Crédits 2025
   Région                                                                      Projet                                 (en MDH)       (en MDH)
                  Entreprise public
                                             Travaux de confortement de la jetée principale du port de commerce
                                                                                                                       126,00         126,00
                 Agence Nationale des        d’Agadir - Tranche 2
                     Ports (ANP)
                                             Aménagement du projet du troisième accès au port d'Agadir                 190,00          20,00

                                             Séparation et transfert des eaux industrielles d'Ait Melloul              186,00          60,50
                 Régie Autonome Multi­
                                             Mise à niveau et extension de la station d'épuration des eaux usées
                   services d'Agadir                                                                                   830,00         300,00
                                             (STEP) M'zar
                       (RAMSA)
                                             Sécurisation de l'alimentation en Eau Potable de la zone Nord d'Agadir    106,00          7,00

Souss - Massa    Agence Nationale des        Reconstitution et mise à niveau générale des régions touchées par le
                                                                                                                       610,00         110,00
                 Eaux et Forêts (ANEF)       séisme d'AI Haouz (Province Taroudant)

                                             Projet Grand Théâtre d'Agadir (Programme de Développement Urbain
                                                                                                                       250,00          4,00
                                             Agadir 2020-2024)

                                             Projet AL Manar                                                           320,92         150,00
                Holding Al Omrane (HAO)      Programme Tiguemi                                                         416,50         100,00

                                             Projet Parc industriel intégré d'Agadir Partie III                        170,00         126,00

                                             Projet Ennour                                                             235,00         120,00

                Investissement 2025 de la Région Guelmim - Oued Noun : 2.260 MDH
                                             Projets d'alimentation en Eau Potable de la zone urbaine                  1.109,98       232,07
  Guelmim-          Office National de       Projets d'alimentation en Eau Potable de la zone rurale                   330,09          34,77
 Oued Noun       l'Electricité et de l'Eau
                     Potable (ONEE)          Projets d'assainissement                                                  201,59          31,61
                                             Renforcement du Réseau 400 kv SUD (Lignes)                                998,60          128,17
PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                    Établissement ou                                                                                      Coût     Crédits 2025
 Région                                                                          Projet                                 (en MDH)     (en MDH)
                    Entreprise public
                                               Contrat de maintenance WARTSILA des centrales diésel Dakhla et
                                                                                                                        448,00        58,38
                                               Tantan
                      Office National de
                   l'Electricité et de l'Eau   Réalisation de la transformation 225/60 KV à Guelmim                      81,86        10,00
                       Potable (ONEE)          Fourniture Pièces de Rechange hors contrat pour la centrale Diesel
                                                                                                                         18,07         5,72
                                               TANTAN
                                               Achèvement des travaux de réalisation de la voie Express Tiznit-
                                                                                                                        373,30
 Guelmim-              Caisse pour le
                                               Laâyoune section : Tiznit-Guelmim
Oued Noun           Financement Routier        Achèvement des travaux de réalisation de la voie Express Tiznit-                       37,08
                           (CFR)               Laâyoune section : - El Ouatia Oued Chbika du PK 1237+400 au PK
                                                                                                                         103,90
                                               1261+188 de Route Nationale 1 - Voie de contournement de la ville d'EL
                                               Ouatia sur une longueur totale de 31.854 KM
                                               Lotissement Oued Noun                                                     122,00         -
                 Holding Al Omrane (HAO)       Mise à niveau urbaine (MANU) Guelmim            Nouveau Programme
                                                                                                                         102,66         -
                                               d'Habitat et d'Urbanisme (NPHU)

                 Investissement 2025 de la Région Laâyoune - Sakia El Hamra : 6.459 MDH
                                               Laverie Laâyoune                                                         2.135,00       5,40
                                               Mine Phosboucraa                                                         550,00        284,75
                           OCP SA              Station d'épuration des eaux usées Al Marsa                              250,00        153,38
                                               Nouveau complexe d’engrais - PHOSBOUCRAA                                 7.180,00     2.500,00
Laâyoune -
 Sakia El                                      Port îlot Laâyoune                                                       5.481,70      716,00
  Hamra                Caisse pour le
                                               Travaux de construction du VIADUC sur Oued Sakia EL Hamra Au
                    Financement Routier                                                                                 1.243,30      378,56
                                               niveau de la voie de contournement de la ville de Laâyoune
                           (CFR)
                   Agence Nationale des
                                               Travaux d'extension du port de Laâyoune                                  150,00        150,00
                       Ports (ANP)
                    Moroccan Agency for        Noor Atlas                                                               768,00        614,00
                   Solar Energy (MASEN)        Projet Nassim Tarfaya Extention                                          808,50        404,25
                                                                                           NOTE SUR LA REPARTITION REGIONALE DE L’INVESTISSEMENT



                      Établissement ou                                                                                     Coût       Crédits 2025
   Région                                                                         Projet                                 (en MDH)       (en MDH)
                      Entreprise public
                                                 Projets d'alimentation en Eau Potable de la zone urbaine                 1.759,70       327,22
                                                 Projets d'alimentation en Eau Potable de la zone rurale                  264,63          16,34
                                                 Projets d'assainissement                                                 397,00          3,04
                                                 Renforcement du Réseau 400 kV SUD (Lignes)                               998,60          198,32
                        Office National de
                     l'Electricité et de l'Eau   Renforcement du Réseau 400 kV SUD (Postes)                               435,22          75,58
                         Potable (ONEE)
                                                 Sécurisation de l'alimentation électrique des Provinces du Sud            49,09          22,11
                                                 Mise à niveau des lignes HTB (Haute Tension Basse)                        98,24          16,00
                                                 Télé-relève des compteurs des clients BT et mise en place d’une
                                                 Infrastructure de Comptage Avancée AMI (Advanced Metering                300,00           1,85
                                                 Infrastructure).
Laâyoune -                                       Achèvement des opérations du Nouveau Programme d'Habitat et
                                                                                                                          260,00          10,00
                                                 d'Urbanisme (NPHU)
 O Cl rX l d L_ 1
  Hamra                                          Projet Madinat 25 Mars Extension bloc R                                  239,66          5,00
                                                 Projet Madinat 25 Mars Ext 2eme Tranche bloc S                           291,66          7,30
                                                 Lotissement Jnane AL Wifaq                                               134,46          30,00
                                                 Lotissement Annasr 1 -Boujdour                                            119,65         6,30
                    Holding Al Omrane (HAO)      Lotissement AL Khair II                                                   114,80         30,30
                                                 Lotissement AL Khair 1                                                    371,65         49,48
                                                 Projet Résidence AL Wiam 1                                               105,33          20,00
                                                 Lotissement Annarjiss                                                     139,81         15,00
                                                 Projet Madinat AL Wifaq Laayoune                                          729,19           -

                                                 Projet Madinat AL Wahda GHIJ 2eme Tranche Laâyoune                        272,71           -
PROJET DE L O! DE FINANCES POUR L'ANNEE2025



                    Établissement ou                                                                                    Coût     Crédits 2025
 Région                                                                        Projet                                 (en MDH)     (en MDH)
                    Entreprise public

                                               Projet Madinat 25 Mars Laâyoune                                        420,35           -
Laâyoune -
                                               Mise à niveau urbaine (MANU) et Nouveau Programme d'Habitat et
 Sakia El        Holding Al Omrane (HAO)                                                                              209,50           -
                                               d'Urbanisme (NPHU) Tranche 2
  Hamra
                                               Lotissement AL Amane - Boujdour                                         221,53          -

                 Investissement 2025 de la Région Dakhla - Oued Ed-Dahab : 1.146 MDH
                                               Projets d'alimentation en Eau Potable de la zone urbaine               1.216,03      256,79

                                               Projets d'alimentation en Eau Potable de la zone rurale                 231,69        17,92

                                               Projets d'assainissement                                                84,83         0,10
                      Office National de
 Dakhla-           l'Electricité et de l'Eau
                                               Deuxième centrale Diesel de DAKHLA                                      396,16        47,18
  Oued                 Potable (ONEE)
Ed-Dahab                                       Contrat de maintenance WARTSILA DAKHLA et TANTAN                       448,00         34,19

                                               Fourniture de pièces de rechange pour les centrales Dakhla et Bir
                                                                                                                        8,46         6,84
                                               guandouz hors contrat
                                               Maintenance systématique du       processus combustible des   unités
                                                                                                                       118,60        4,00
                                               turbines à gaz et diesel

                                               Lotissement AN Nahda II- Dakhla                                        274,00        20,00
                 Holding Al Omrane (HAO)
                                               Lotissement AL Mohit Dakhla                                             120,38          -
  
"""
    conversation_history = StreamlitChatMessageHistory()  # Créez l'instance pour l'historique

    st.header("PLF2025: Explorez le rapport note sur la répartition régionale de l'investissement à travers notre chatbot💬")
    
    # Load the document
    #docx = 'PLF2025-Rapport-FoncierPublic_Fr.docx'
    
    #if docx is not None:
        # Lire le texte du document
        #text = docx2txt.process(docx)
        #with open("so.txt", "w", encoding="utf-8") as fichier:
            #fichier.write(text)

        # Afficher toujours la barre de saisie
    st.markdown('<div class="input-space"></div>', unsafe_allow_html=True)
    selected_questions = st.sidebar.radio("****Choisir :****", questions)
        # Afficher toujours la barre de saisie
    query_input = st.text_input("", key="text_input_query", placeholder="Posez votre question ici...", help="Posez votre question ici...")
    st.markdown('<div class="input-space"></div>', unsafe_allow_html=True)

    if query_input and query_input not in st.session_state.previous_question:
        query = query_input
        st.session_state.previous_question.append(query_input)
    elif selected_questions:
        query = selected_questions
    else:
        query = ""

    if query :
        st.session_state.conversation_history.add_user_message(query) 
        if "Donnez-moi un résumé du rapport" in query:
            summary="""Le document présente une répartition régionale des investissements prévue dans le cadre du projet de loi de finances pour l'année budgétaire 2025. Il souligne les objectifs principaux, notamment l'amélioration des infrastructures à travers le désenclavement routier et l'accès à l'eau potable et à l'électrification. Des programmes spécifiques, tels que "Ville sans Bidonvilles" (VSB) et "Habitat Menaçant Ruine" (HMR), sont également mentionnés pour la restructuration urbaine et la mise à niveau des quartiers sous-équipés. Les données chiffrées sur les investissements passés et futurs sont également fournies pour les années 2021 à 2025."""
            st.session_state.conversation_history.add_ai_message(summary) 

        else:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{query}. Répondre à la question d'apeés ce texte repondre justement à partir de texte ne donne pas des autre information voila le texte donnee des réponse significatif et bien formé essayer de ne pas dire que information nest pas mentionné dans le texte si tu ne trouve pas essayer de repondre dapres votre connaissance ms focaliser sur ce texte en premier: {text} "
                    )
                }
            ]

            # Appeler l'API OpenAI pour obtenir le résumé
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages
            )

            # Récupérer le contenu de la réponse

            summary = response['choices'][0]['message']['content']
           
                # Votre logique pour traiter les réponses
            #conversation_history.add_user_message(query)
            #conversation_history.add_ai_message(response)
            st.session_state.conversation_history.add_ai_message(summary)  # Ajouter à l'historique
            
            # Afficher la question et le résumé de l'assistant
            #conversation_history.add_user_message(query)
            #conversation_history.add_ai_message(summary)

            # Format et afficher les messages comme précédemment
                
            # Format et afficher les messages comme précédemment
        formatted_messages = []
        previous_role = None 
        if st.session_state.conversation_history.messages: # Variable pour stocker le rôle du message précédent
                for msg in conversation_history.messages:
                    role = "user" if msg.type == "human" else "assistant"
                    avatar = "🧑" if role == "user" else "🤖"
                    css_class = "user-message" if role == "user" else "assistant-message"

                    if role == "user" and previous_role == "assistant":
                        message_div = f'<div class="{css_class}" style="margin-top: 25px;">{msg.content}</div>'
                    else:
                        message_div = f'<div class="{css_class}">{msg.content}</div>'

                    avatar_div = f'<div class="avatar">{avatar}</div>'
                
                    if role == "user":
                        formatted_message = f'<div class="message-container user"><div class="message-avatar">{avatar_div}</div><div class="message-content">{message_div}</div></div>'
                    else:
                        formatted_message = f'<div class="message-container assistant"><div class="message-content">{message_div}</div><div class="message-avatar">{avatar_div}</div></div>'
                
                    formatted_messages.append(formatted_message)
                    previous_role = role  # Mettre à jour le rôle du message précédent

                messages_html = "\n".join(formatted_messages)
                st.markdown(messages_html, unsafe_allow_html=True)
if __name__ == '__main__':
    main()
