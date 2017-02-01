#pragma once

#include <Pantheon/error.h>
#include <Pantheon/stddef.h>

namespace Pantheon
{
    namespace Container
    {
        struct LinkedList
        {
            struct Node
            {
                BYTE *Data;
                Node *Next;
            };

            QWORD ElementSize;
            Node *HeadNode, *TailNode;

            static PRESULT Create(retval LinkedList *pLinkedList, QWORD ElementSize);

            static PRESULT Dispose(LinkedList *pLinkedList);

            static PRESULT GetFront(retval Node *pNode, const LinkedList *LinkedList);
            static PRESULT GetBack(retval Node *pNode, const LinkedList *LinkedList);
            static PRESULT IsEmpty(const LinkedList *LinkedList);

            static PRESULT PushBack(retval nullable Node *pNode, LinkedList *LinkedList, const BYTE *Data);
            static PRESULT Update(LinkedList *LinkedList, const Node *pNode, const BYTE *pData);
            static PRESULT Remove(LinkedList *LinkedList, Node *pNode);

            static Node *Release(LinkedList *LinkedList, Node *pNode);
            static PRESULT PushFront(LinkedList *LinkedList, Node *pNode);
            static PRESULT PushBack(LinkedList *LinkedList, Node *pNode);
        };
    }
}