# from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text,Boolean,Date,FLOAT
# from sqlalchemy.orm import declarative_base, relationship

# Base = declarative_base()

# class ParamEtiketlers(Base):
#     __tablename__ = "ParamEtiketlers"
#     ID = Column(Integer,primary_key=True)
#     TagName = Column(String)

# class InfoofUrls(Base):
#     __tablename__ = "InfoofUrls"
#     ID = Column(Integer,primary_key=True)
#     Url = Column(String)
#     codeofUrl = Column(String)

# class UrlEtiketlers(Base):
#     __tablename__ = "UrlEtiketlers"
#     ID = Column(Integer,primary_key=True)
#     UrlID = Column(Integer)
#     TagID = Column(Integer)

# class UrlOzet(Base):
#     __tablename__ = "UrlOzet"
#     ID = Column(Integer,primary_key=True)
#     Url = Column(String)
#     OzetUrl = Column(String)
#     BodyVarmi = Column(Boolean)

# class TakipTablo(Base):
#     __tablename__ = "TakipTablo"
#     ID = Column(Integer,primary_key=True)
#     Sure = Column(FLOAT,nullable=True)
#     Sayac = Column(Integer,nullable=True)


# # --- Veritabanı oluşturma kısmı ---
# # SQLite kullanıyorsan:
# engine = create_engine("mssql+pyodbc://@localhost\\SQLEXPRESS/VeriProjesi?"
#     "driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes")

# # Tabloları oluştur
# Base.metadata.create_all(engine)