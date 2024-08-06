#ifndef __JSON_H__
#define __JSON_H__


#include <stdio.h>
#include <string.h>
#include "common.h"
#include "cJSON.h"

char* strdup(const char* s);
/* JSON */
/* root is the root of the JSON object */
/* key is of the format ".key1.key2.key3..." or "key1.key2.key3..." */
static inline cJSON* JSONGetItem(cJSON* root, const char* key) {
	
	const char delim[] = ".";
	cJSON* item = root;
	char* token = NULL;
	char* keyDup = strdup(key);
	if (keyDup && keyDup[0] == '.') {
		keyDup++;
	}
	token = strtok(keyDup, delim);
	
	while (token != NULL) {
		item = cJSON_GetObjectItem(item, token);
		if (item == NULL) {
			break;
		}
		token = strtok(NULL, delim);
	}
	
	free(keyDup);
	return item;
}

#endif /* __JSON_H__ */
