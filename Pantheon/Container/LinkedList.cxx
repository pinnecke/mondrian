#include <Pantheon/Container/LinkedList.h>

namespace Pantheon
{
    namespace Container
    {
        PRESULT LinkedList::Create(retval LinkedList *pLinkedList, QWORD ElementSize)
        {
            if ((pLinkedList == nullptr) || (ElementSize < 1))
                return PRESULT::IllegalArgument;
            pLinkedList->ElementSize = ElementSize;
            pLinkedList->HeadNode = pLinkedList->TailNode = nullptr;
            return PRESULT::OK;
        }

        PRESULT LinkedList::Dispose(LinkedList *pLinkedList)
        {
            if (pLinkedList == nullptr)
                return PRESULT::IllegalArgument;
            Node *it = pLinkedList->HeadNode;
            while (it != nullptr) {
                Node *subject = it;
                it = it->Next;

                free (subject->Data);
                free (subject);
            }
            return PRESULT::OK;
        }

        PRESULT LinkedList::GetFront(retval Node *pNode, const LinkedList *pLinkedList)
        {
            if ((pNode == nullptr) || (pLinkedList == nullptr))
                return PRESULT::IllegalArgument;

            if (pLinkedList->HeadNode == nullptr)
                return PRESULT::NoSuchElement;

            *pNode = *pLinkedList->HeadNode;

            return PRESULT::OK;
        }

        PRESULT LinkedList::GetBack(retval Node *pNode, const LinkedList *pLinkedList)
        {
            if ((pNode == nullptr) || (pLinkedList == nullptr))
                return PRESULT::IllegalArgument;

            if (pLinkedList->TailNode == nullptr)
                return PRESULT::NoSuchElement;

            *pNode = *pLinkedList->TailNode;

            return PRESULT::OK;
        }

        PRESULT LinkedList::IsEmpty(const LinkedList *pLinkedList)
        {
            if (pLinkedList == nullptr)
                return PRESULT::IllegalArgument;
            return (pLinkedList->HeadNode == nullptr)? PRESULT::True : PRESULT::False;
        }

        PRESULT LinkedList::PushBack(retval nullable Node *pNode, LinkedList *pLinkedList, const BYTE *pData)
        {
            if ((pNode == nullptr) || (pLinkedList == nullptr) || (pData == nullptr))
                return PRESULT::IllegalArgument;

            return PRESULT::OK;
        }

        PRESULT LinkedList::Update(LinkedList *LinkedList, const Node *Node, const BYTE *pData)
        {
            // TODO
            return PRESULT::Failed;
        }

        PRESULT LinkedList::Remove(LinkedList *LinkedList, Node *Node)
        {
            // TODO
            return PRESULT::Failed;
        }

        Node *LinkedList::Release(LinkedList *LinkedList, Node *Node)
        {
            // TODO
            return PRESULT::Failed;
        }

        PRESULT LinkedList::PushFront(LinkedList *LinkedList, Node *Node)
        {
            // TODO
            return PRESULT::Failed;
        }

        PRESULT LinkedList::PushBack(LinkedList *LinkedList, Node *Node)
        {
            // TODO
            return PRESULT::Failed;
        }
    }
}