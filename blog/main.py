from fastapi import FastAPI, Depends, status, Response, HTTPException
import models, schemas
from database import engine, SessionLocal
from sqlalchemy.orm import Session

app = FastAPI()

models.Base.metadata.create_all(bind=engine)

def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()

# Create a Blog

@app.post('/blogs', status_code=status.HTTP_201_CREATED) #status_code=201 also works
def create(blog:schemas.Blog, db:Session = Depends(get_db)):
	new_blog = models.Blog(title=blog.title, body=blog.body)
	db.add(new_blog)
	db.commit()
	db.refresh(new_blog)
	return new_blog

# Delete a Blog

@app.delete('/blogs/{id}', status_code=status.HTTP_204_NO_CONTENT)
def destroy(id:int, db:Session=Depends(get_db)):
	blog = db.query(models.Blog).filter(models.Blog.id == id)
	if not blog.first():
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
							detail=f'Blog {id} not found.')
	blog.delete(synchronize_session=False)
	db.commit()
	return {'detail':f'Blog {id} has been deleted.'}

# Update a Blog

@app.put('/blog/{id}', status_code=status.HTTP_202_ACCEPTED)
def update(id:int, blog:schemas.Blog, db:Session = Depends(get_db)):
	blog_ = db.query(models.Blog).filter(models.Blog.id == id)
	if not blog_.first():
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
							detail=f'Blog {id} not found.')
	blog_.update(blog, synchronize_session=False)
	db.commit()
	return f'Blog {id} updated successfully.'

# Show all Blogs

@app.get('/blogs')
def all(db:Session=Depends(get_db)):
	blogs = db.query(models.Blog).all()
	return blogs

# Show a particular Blog
@app.get('/blogs/{id}', status_code=200)
def show(id:int, response:Response, db:Session=Depends(get_db)):
	# blog = db.query(models.Blog)[id-1]
	blog = db.query(models.Blog).filter(models.Blog.id == id).first()
	
	if not blog:
		#response.status_code = status.HTTP_404_NOT_FOUND
		#return {'detail':f'Blog with ID {id} not found'}
		
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
							detail=f'Blog with ID {id} not found')

	
	return blog