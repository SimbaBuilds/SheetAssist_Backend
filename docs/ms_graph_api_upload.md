# Upload or replace the contents of a driveItem

**Article** | **02/08/2024** | **12 contributors**

## Feedback

### In this article
1. Permissions
2. HTTP request
3. Request headers
4. Request body

*Namespace: microsoft.graph*

Provide the contents of a new file or update the contents of an existing file in a single API call. This method only supports files up to 250 MB in size.

To upload large files, see Upload large files with an upload session.

This API is available in the following national cloud deployments.

### National Cloud Deployments
| Service | Availability |
|---------|-------------|
| Global service | ✅ |
| US Government L4 | ✅ |
| US Government L5 (DOD) | ✅ |
| China operated by 21Vianet | ✅ |

### Permissions

Choose the permission or permissions marked as least privileged for this API. Use a higher privileged permission or permissions only if your app requires it. For details about delegated and application permissions, see Permission types. To learn more about these permissions, see the permissions reference.

| Permission type | Least privileged permissions | Higher privileged permissions |
|----------------|--------------------------|---------------------------|
| Delegated (work or school account) | Files.ReadWrite | Files.ReadWrite.All, Sites.ReadWrite.All |
| Delegated (personal Microsoft account) | Files.ReadWrite | Files.ReadWrite.All |
| Application | Files.ReadWrite.All | Sites.ReadWrite.All |

### HTTP request

#### To replace an existing item
```http
PUT /drives/{drive-id}/items/{item-id}/content
PUT /groups/{group-id}/drive/items/{item-id}/content
PUT /me/drive/items/{item-id}/content
PUT /sites/{site-id}/drive/items/{item-id}/content
PUT /users/{user-id}/drive/items/{item-id}/content
```

#### To upload a new file
```http
PUT /drives/{drive-id}/items/{parent-id}:/{filename}:/content
PUT /groups/{group-id}/drive/items/{parent-id}:/{filename}:/content
PUT /me/drive/items/{parent-id}:/{filename}:/content
PUT /sites/{site-id}/drive/items/{parent-id}:/{filename}:/content
PUT /users/{user-id}/drive/items/{parent-id}:/{filename}:/content
```

### Request headers

| Name | Description |
|------|-------------|
| Authorization | Bearer {token}. Required. Learn more about authentication and authorization. |
| Content-Type | text/plain. Required. |

### Request body

The contents of the request body should be the binary stream of the file to be uploaded.

### Response

If successful, this method returns a driveItem object in the response body for the newly created or updated file.

### Example (upload a new file)

This example uploads the string "The contents of the file goes here." to a file in the signed-in user's drive under FolderA named FileB.txt.

```http
PUT https://graph.microsoft.com/v1.0/me/drive/root:/FolderA/FileB.txt:/content
Content-Type: text/plain

The contents of the file goes here.
```

#### Response

If successful, this method returns an driveItem resource in the response body for the newly created file.

```http
HTTP/1.1 201 Created
Content-Type: application/json

{
  "id": "0123456789abc",
  "name": "FileB.txt",
  "size": 35,
  "file": { }
}
```

### Example (updating an existing file)

This example replaces the contents of a file with a known ID.

```http
PUT https://graph.microsoft.com/v1.0/me/drive/items/{item-id}/content
Content-Type: text/plain

The contents of the file goes here.
```

#### Response

If successful, this method returns an driveItem resource in the response body for the newly created file.

```http
HTTP/1.1 201 Created
Content-Type: application/json

{
  "id": "0123456789abc",
  "name": "FileB.txt",
  "size": 35,
  "file": { }
}
```