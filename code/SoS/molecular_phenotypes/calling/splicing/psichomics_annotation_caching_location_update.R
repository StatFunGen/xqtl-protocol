moveFiles <- function(package){
  olddir <- path.expand(rappdirs::user_cache_dir(appname=package))
  newdir <- tools::R_user_dir(package, which="cache")
  dir.create(path=newdir, recursive=TRUE)
  files <- list.files(olddir, full.names =TRUE)
  moveres <- vapply(files,
                    FUN=function(fl){
                      filename = basename(fl)
                      newname = file.path(newdir, filename)
                      file.rename(fl, newname)
                    },
                    FUN.VALUE = logical(1))
  if(all(moveres)) unlink(olddir, recursive=TRUE)
}

package="AnnotationHub"

moveFiles(package)